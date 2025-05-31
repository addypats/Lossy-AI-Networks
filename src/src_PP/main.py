# main.py

import os
import yaml
import random
import torch
import argparse

from comms import LossyNetwork
from data import get_dataset
from models import get_classifier_and_tokenizer
from trainer import MyClassifierCallback, compute_classfication_metrics

import wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


# ============================================
# 1) Helper: Zero out a fraction of a Tensor
# ============================================
def zero_some_fraction(tensor: torch.Tensor, percent: float) -> torch.Tensor:
    """
    Randomly zero out `percent` fraction of elements in `tensor`.
    Returns a new tensor (same device & dtype) with that fraction zeroed.
    """
    if percent <= 0.0:
        return tensor
    mask = (torch.rand_like(tensor) >= percent).to(tensor.dtype)
    return tensor * mask


# ============================================
# 2.A) Stage0: Embedding + transformer blocks
# ============================================
class Stage0(nn.Module):
    """
    Stage 0: holds the embedding layer + a contiguous slice of transformer blocks.
    Expects input: (input_ids: LongTensor, attention_mask: LongTensor) on cuda:0
    Returns: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:0
    """
    def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
        """
        full_model:  AutoModelForSequenceClassification loaded on CPU
        layer_start/layer_end: indices into full_model.model.layers
        device:      e.g. "cuda:0"
        """
        super().__init__()
        self.device = torch.device(device)

        # Embedding from the base model
        self.embed = full_model.model.embed_tokens.to(self.device)

        # A slice of transformer blocks [layer_start:layer_end]
        self.layers = nn.ModuleList(
            [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
        )

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        input_ids:      LongTensor [batch, seq_len] on cuda:0
        attention_mask: LongTensor [batch, seq_len] on cuda:0
        """
        # (1) Embed tokens → [batch, seq_len, hidden_dim]
        h = self.embed(input_ids)  # -> (batch, seq_len, hidden_dim)

        # (2) Build position IDs: 0..(seq_len-1) for every batch
        batch_size, seq_len = input_ids.size()
        position_ids = (
            torch.arange(seq_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )  # [batch, seq_len] long

        # (3) Pass through each transformer block, supplying position_ids
        for block in self.layers:
            h = block(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        # (4) Return hidden states + attention mask for the next stage
        return h, attention_mask


# ============================================
# 2.B) StageMiddle: Intermediate transformer blocks
# ============================================
class StageMiddle(nn.Module):
    """
    Stage k where 1 ≤ k ≤ N-2.  Holds a contiguous slice of transformer blocks.
    Expects input: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:k
    Returns: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:k
    """
    def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.layers = nn.ModuleList(
            [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
        )

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor):
        """
        hidden_states:  FloatTensor [batch, seq_len, hidden_dim] on cuda:k
        attention_mask: LongTensor [batch, seq_len] on cuda:k
        """
        h = hidden_states  # [batch, seq_len, hidden_dim]
        batch_size, seq_len, _ = h.size()

        # (1) Build position IDs
        position_ids = (
            torch.arange(seq_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )  # [batch, seq_len] long

        # (2) Pass through each transformer block, supplying position_ids
        for block in self.layers:
            h = block(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        return h, attention_mask


# ============================================
# 2.C) StageLast: Final transformer blocks + norm + classification head
# ============================================
class StageLast(nn.Module):
    """
    Stage N-1: holds a contiguous slice of transformer blocks,
    plus final LayerNorm and classification head.
    Expects input: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:N-1
    If labels are given, returns scalar loss; otherwise returns logits.
    """
    def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
        super().__init__()
        self.device = torch.device(device)
        self.layers = nn.ModuleList(
            [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
        )
        self.final_norm = full_model.model.norm.to(self.device)
        self.classifier = full_model.score.to(self.device)

    def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor, labels=None):
        """
        hidden_states:  FloatTensor [batch, seq_len, hidden_dim] on cuda:N-1
        attention_mask: LongTensor [batch, seq_len] on cuda:N-1
        labels:         LongTensor [batch] on cuda:N-1 (optional)
        """
        h = hidden_states  # [batch, seq_len, hidden_dim]
        batch_size, seq_len, _ = h.size()

        # (1) Build position IDs
        position_ids = (
            torch.arange(seq_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )  # [batch, seq_len] long

        # (2) Pass through each transformer block, supplying position_ids
        for block in self.layers:
            h = block(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        # (3) Final norm + classification head
        h_norm = self.final_norm(h)                 # [batch, seq_len, hidden_dim]
        cls_token = h_norm[:, 0, :]                 # [batch, hidden_dim]
        logits = self.classifier(cls_token)         # [batch, num_labels]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits, labels)

        return logits


# ============================================
# 3) Main Training Function
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline Parallel Classification with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2,
                        help='Number of GPUs to use (i.e. number of pipeline stages)')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate (0.0–1.0)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Model name (checkpoint for AutoModelForSequenceClassification)')
    parser.add_argument('--batch_size', type=int, default=64, help='Global batch size (must divide by num_nodes)')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision (FP16)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', '-d', type=str, default='winogrande', help='Dataset to use')
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Maximum number of training samples to use (0 = all)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--eval_steps', type=int, default=50, help='(Optional) steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every N global steps')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log to WandB every N global steps')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--run_id', type=str, required=True, help='Unique run identifier')
    parser.add_argument('-nunf', '--num_unfrozen_layers', type=int, default=None,
                        help='Number of last transformer layers to unfreeze (others are frozen)')
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────
    # 3.1 Set Random Seeds
    # ──────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ──────────────────────────────────────────────────────────
    # 3.2 Load dataset_config.yaml
    # ──────────────────────────────────────────────────────────
    with open("src/src_PP/dataset_config.yaml") as config_file:
        dataset_config_full = yaml.safe_load(config_file)
    dataset_config = dataset_config_full[args.dataset]

    # ──────────────────────────────────────────────────────────
    # 3.3 Initialize LossyNetwork
    # ──────────────────────────────────────────────────────────
    network = LossyNetwork(args)
    network.set_seed(args.seed)

    # ──────────────────────────────────────────────────────────
    # 3.4 Load classification model + tokenizer via models.py
    # ──────────────────────────────────────────────────────────
    model_cls, tokenizer = get_classifier_and_tokenizer(
        args.model_name,
        num_labels=dataset_config["num_labels"],
        num_unfrozen_layers=args.num_unfrozen_layers
    )
    # `model_cls` is AutoModelForSequenceClassification (e.g. LlamaForSequenceClassification),
    # possibly with early layers frozen if `nunf` is specified.

    # ──────────────────────────────────────────────────────────
    # 3.5 Load train & eval datasets
    # ──────────────────────────────────────────────────────────
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

    # ──────────────────────────────────────────────────────────
    # 3.6 Prepare output directory, save args.yaml
    # ──────────────────────────────────────────────────────────
    OUTPUT_DIR_LOCAL = f"{args.output_dir}/{args.run_id}"
    os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)
    with open(f"{OUTPUT_DIR_LOCAL}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # ──────────────────────────────────────────────────────────
    # 3.7 Initialize WandB
    # ──────────────────────────────────────────────────────────
    wandb.init(
        project="your_project_name",
        name=args.run_id,
        reinit=True
    )
    wandb.config.update({
        "dataset": args.dataset,
        "run_id": args.run_id,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "loss_rate": args.loss_rate,
        "learning_rate": args.learning_rate,
        "num_unfrozen_layers": args.num_unfrozen_layers,
        "num_nodes": args.num_nodes,
        "fp16": args.fp16,
    })

    # ──────────────────────────────────────────────────────────
    # 3.8 Pipeline Hyperparameters (from args)
    # ──────────────────────────────────────────────────────────
    NUM_GPUS = args.num_nodes
    TOTAL_LAYERS = model_cls.model.config.num_hidden_layers  # e.g. 32 for Llama-3.2-1B
    assert TOTAL_LAYERS % NUM_GPUS == 0, f"Total layers ({TOTAL_LAYERS}) must be divisible by num_nodes"
    LAYERS_PER_STAGE = TOTAL_LAYERS // NUM_GPUS

    GLOBAL_BATCH_SIZE = args.batch_size
    MICRO_BATCHES = NUM_GPUS   # Often one micro-batch per GPU
    assert GLOBAL_BATCH_SIZE % MICRO_BATCHES == 0, "Global batch size must be divisible by num_nodes"
    MICRO_BATCH_SIZE = GLOBAL_BATCH_SIZE // MICRO_BATCHES

    LOSSY_FRACTION = args.loss_rate
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    SAVE_EVERY_N_STEPS = args.save_steps
    LOG_EVERY_N_STEPS = args.logging_steps
    MAX_SEQ_LEN = args.max_length

    DEVICE_LIST = [f"cuda:{i}" for i in range(NUM_GPUS)]
    for dev in DEVICE_LIST:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{dev} is not available")

    # ──────────────────────────────────────────────────────────
    # 3.9 Handle Single‐GPU (NUM_GPUS == 1)
    # ──────────────────────────────────────────────────────────
    if NUM_GPUS == 1:
        # Standard single‐GPU fine‐tuning with model_cls
        device = torch.device("cuda:0")
        model_cls.to(device)
        optimizer = optim.AdamW(model_cls.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scaler = GradScaler() if args.fp16 else None

        # DataLoader (shared collate_fn)
        def collate_fn(batch):
            input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
            attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
            labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        train_loader = DataLoader(
            train_dataset,
            batch_size=GLOBAL_BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        callback_args = {
            "report_ttac": dataset_config["report_ttac"],
            "report_file": f"{OUTPUT_DIR_LOCAL}/ttac_report.txt",
            "target_acc": dataset_config["target_acc"],
        }
        callback = MyClassifierCallback(callback_args)

        global_step = 0
        for epoch in range(NUM_EPOCHS):
            print(f"[Single‐GPU] Epoch {epoch+1}/{NUM_EPOCHS} starting...")
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                if args.fp16:
                    with autocast():
                        outputs = model_cls(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model_cls(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                global_step += 1
                if global_step % LOG_EVERY_N_STEPS == 0:
                    wandb.log({
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": loss.item()
                    })

                if global_step % SAVE_EVERY_N_STEPS == 0:
                    ckpt = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "model_state": model_cls.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }
                    ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, f"checkpoint-step{global_step}.pt")
                    torch.save(ckpt, ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")

            print(f"[Single‐GPU] Epoch {epoch+1}/{NUM_EPOCHS} complete.")
        print("Single‐GPU training finished!")
        wandb.finish()
        return

    # ──────────────────────────────────────────────────────────
    # 3.10 Instantiate Each Pipeline Stage (NUM_GPUS ≥ 2)
    # ──────────────────────────────────────────────────────────
    stages = []
    for i in range(NUM_GPUS):
        start_idx = i * LAYERS_PER_STAGE
        end_idx = start_idx + LAYERS_PER_STAGE
        is_last = (i == NUM_GPUS - 1)
        if i == 0:
            stage = Stage0(
                full_model=model_cls,
                layer_start=start_idx,
                layer_end=end_idx,
                device=DEVICE_LIST[i]
            )
        elif is_last:
            stage = StageLast(
                full_model=model_cls,
                layer_start=start_idx,
                layer_end=end_idx,
                device=DEVICE_LIST[i]
            )
        else:
            stage = StageMiddle(
                full_model=model_cls,
                layer_start=start_idx,
                layer_end=end_idx,
                device=DEVICE_LIST[i]
            )
        stages.append(stage)

    # ──────────────────────────────────────────────────────────
    # 3.11 Create One Optimizer per Stage
    # ──────────────────────────────────────────────────────────
    optimizers = []
    for st in stages:
        opt = optim.AdamW(st.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        optimizers.append(opt)

    # ──────────────────────────────────────────────────────────
    # 3.12 Prepare DataLoader (no DistributedSampler)
    # ──────────────────────────────────────────────────────────
    def collate_fn(batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
        attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
        labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(
        train_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # ──────────────────────────────────────────────────────────
    # 3.13 Setup Mixed Precision if Requested
    # ──────────────────────────────────────────────────────────
    scaler = GradScaler() if args.fp16 else None

    # ──────────────────────────────────────────────────────────
    # 3.14 Create TTAC Callback (unchanged)
    # ──────────────────────────────────────────────────────────
    callback_args = {
        "report_ttac": dataset_config["report_ttac"],
        "report_file": f"{OUTPUT_DIR_LOCAL}/ttac_report.txt",
        "target_acc": dataset_config["target_acc"],
    }
    callback = MyClassifierCallback(callback_args)

    # ──────────────────────────────────────────────────────────
    # 3.15 Training Loop: 2*K – 1 Pipeline Schedule
    # ──────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} starting...")
        for batch in train_loader:
            # (a) Split the global batch into MICRO_BATCHES micro-batches
            input_chunks = batch["input_ids"].chunk(MICRO_BATCHES)
            mask_chunks = batch["attention_mask"].chunk(MICRO_BATCHES)
            label_chunks = batch["labels"].chunk(MICRO_BATCHES)

            # Buffers for activations & masks at each stage
            buf_h = [[None] * MICRO_BATCHES for _ in range(NUM_GPUS)]
            buf_m = [[None] * MICRO_BATCHES for _ in range(NUM_GPUS)]

            # Buffers for final losses and upstream gradients
            buf_loss = [None] * MICRO_BATCHES
            grad_buf = [[None] * MICRO_BATCHES for _ in range(NUM_GPUS - 1)]

            # Zero all optimizers
            for opt in optimizers:
                opt.zero_grad()

            total_steps = 2 * MICRO_BATCHES - 1
            for t in range(total_steps):
                # ───────────────────────────────────────────────────
                # (1) Stage 0 Forward: t < MICRO_BATCHES
                # ───────────────────────────────────────────────────
                if t < MICRO_BATCHES:
                    i = t
                    x0 = input_chunks[i].to(DEVICE_LIST[0])    # [batch, seq_len] LongTensor
                    m0 = mask_chunks[i].to(DEVICE_LIST[0])     # [batch, seq_len] LongTensor

                    if args.fp16:
                        with autocast():
                            h0, a0 = stages[0](x0, m0)
                    else:
                        h0, a0 = stages[0](x0, m0)

                    buf_h[0][i] = h0.detach()
                    buf_m[0][i] = a0.detach()

                # ───────────────────────────────────────────────────
                # (2) Stage k Forward for 1 ≤ k ≤ N-2:
                #     Condition: k ≤ t < MICRO_BATCHES + k
                # ───────────────────────────────────────────────────
                for k in range(1, NUM_GPUS - 1):
                    if k <= t < (MICRO_BATCHES + k):
                        i = t - k
                        h_in = buf_h[k - 1][i].to(DEVICE_LIST[k]).requires_grad_(True)
                        m_in = buf_m[k - 1][i].to(DEVICE_LIST[k])
                        if args.fp16:
                            with autocast():
                                h_out, a_out = stages[k](h_in, m_in)
                        else:
                            h_out, a_out = stages[k](h_in, m_in)
                        buf_h[k][i] = h_out.detach()
                        buf_m[k][i] = a_out.detach()

                # ───────────────────────────────────────────────────
                # (3) Stage N-1 (Last) Forward + Loss:
                #     Condition: (N-1) ≤ t < MICRO_BATCHES + (N-1)
                # ───────────────────────────────────────────────────
                last_stage = NUM_GPUS - 1
                if (NUM_GPUS - 1) <= t < (MICRO_BATCHES + NUM_GPUS - 1):
                    i = t - (NUM_GPUS - 1)
                    h_last_in = buf_h[last_stage - 1][i].to(DEVICE_LIST[last_stage]).requires_grad_(True)
                    m_last_in = buf_m[last_stage - 1][i].to(DEVICE_LIST[last_stage])
                    lbl = label_chunks[i].to(DEVICE_LIST[last_stage])

                    if args.fp16:
                        with autocast():
                            loss_i = stages[last_stage](h_last_in, m_last_in, labels=lbl)
                        buf_loss[i] = loss_i
                    else:
                        loss_i = stages[last_stage](h_last_in, m_last_in, labels=lbl)
                        buf_loss[i] = loss_i

                # ───────────────────────────────────────────────────
                # (4) Stage N-1 Backward: t ≥ (MICRO_BATCHES + (N-1))
                # ───────────────────────────────────────────────────
                if t >= (MICRO_BATCHES + NUM_GPUS - 1):
                    i = t - (MICRO_BATCHES + NUM_GPUS - 1)
                    if args.fp16:
                        scaler.scale(buf_loss[i]).backward(retain_graph=True)
                    else:
                        buf_loss[i].backward(retain_graph=True)
                    # Capture gradient wrt h_last_in on cuda:(N-1)
                    g_last = h_last_in.grad
                    gz = zero_some_fraction(g_last, LOSSY_FRACTION)
                    grad_buf[last_stage - 1][i] = gz.to(DEVICE_LIST[last_stage - 1])

                # ───────────────────────────────────────────────────
                # (5) Backward through Stages N-2 down to 1:
                #     For each k: if t ≥ (MICRO_BATCHES + k)
                # ───────────────────────────────────────────────────
                for k in range(NUM_GPUS - 2, 0, -1):
                    if t >= (MICRO_BATCHES + k):
                        i = t - (MICRO_BATCHES + k)
                        # Recompute forward for stage k
                        h_in_rec = buf_h[k - 1][i].to(DEVICE_LIST[k]).requires_grad_(True)
                        m_in_rec = buf_m[k - 1][i].to(DEVICE_LIST[k])
                        if args.fp16:
                            with autocast():
                                h_out_rec, _ = stages[k](h_in_rec, m_in_rec)
                        else:
                            h_out_rec, _ = stages[k](h_in_rec, m_in_rec)
                        # Backprop through stage k
                        h_out_rec.backward(grad_buf[k][i], retain_graph=True)
                        gk = h_in_rec.grad
                        gkz = zero_some_fraction(gk, LOSSY_FRACTION)
                        grad_buf[k - 1][i] = gkz.to(DEVICE_LIST[k - 1])

                # ───────────────────────────────────────────────────
                # (6) Backward through Stage 0: t ≥ MICRO_BATCHES
                # ───────────────────────────────────────────────────
                if t >= MICRO_BATCHES:
                    i = t - MICRO_BATCHES
                    x0_rec = input_chunks[i].to(DEVICE_LIST[0]).requires_grad_(True)
                    m0_rec = mask_chunks[i].to(DEVICE_LIST[0])
                    if args.fp16:
                        with autocast():
                            h0_rec, _ = stages[0](x0_rec, m0_rec)
                            h0_rec.backward(grad_buf[0][i], retain_graph=True)
                    else:
                        h0_rec, _ = stages[0](x0_rec, m0_rec)
                        h0_rec.backward(grad_buf[0][i], retain_graph=True)

            # ──────────────────────────────────────────────────────────
            # (b) Optimizer Step: update each stage’s parameters
            # ──────────────────────────────────────────────────────────
            if args.fp16:
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in optimizers:
                    opt.step()

            # ──────────────────────────────────────────────────────────
            # (c) Logging & Checkpointing
            # ──────────────────────────────────────────────────────────
            global_step += 1
            micro_losses = [buf_loss[i].item() for i in range(MICRO_BATCHES)]
            avg_loss = sum(micro_losses) / MICRO_BATCHES

            if global_step % LOG_EVERY_N_STEPS == 0:
                wandb.log({
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": avg_loss
                })

            if global_step % SAVE_EVERY_N_STEPS == 0:
                ckpt = {
                    "global_step": global_step,
                    "epoch": epoch,
                }
                for i in range(NUM_GPUS):
                    ckpt[f"stage{i}_state"] = stages[i].state_dict()
                    ckpt[f"optimizer{i}_state"] = optimizers[i].state_dict()
                ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, f"checkpoint-step{global_step}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        # End of epoch
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} complete.")

        # ───────────────────────────────────────────────────────
        # (Optional) Evaluation & Callback invocation here
        # You can run an evaluation loop over eval_loader, call
        # compute_classfication_metrics, then invoke callback.on_evaluate(...)
        # exactly as before.
        # ───────────────────────────────────────────────────────

    print("Training finished!")
    wandb.finish()


if __name__ == "__main__":
    main()
