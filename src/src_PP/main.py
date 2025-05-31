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
from transformers import LlamaForCausalLM, LlamaTokenizer
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
    `percent` ∈ [0.0, 1.0). Returns a new tensor (same device & dtype) with
    that fraction zeroed.
    """
    if percent <= 0.0:
        return tensor
    mask = (torch.rand_like(tensor) >= percent).to(tensor.dtype)
    return tensor * mask


# ============================================
# 2) Define a Dynamic Number of Pipeline Stages
#    Each stage holds a contiguous slice of Llama’s transformer blocks.
# ============================================
class PipelineStage(nn.Module):
    """
    A generic “stage” that holds `layers_per_stage` transformer blocks
    (or the final norm + LM head if it’s the last stage).
    
    We will instantiate as many of these as there are GPUs (num_nodes),
    with each stage placed on cuda:stage_rank.
    """
    def __init__(self, full_llama, layer_start: int, layer_end: int,
                 is_last_stage: bool, device: str):
        """
        full_llama:      the original LlamaForCausalLM loaded on CPU
        layer_start:     inclusive start index in full_llama.model.layers
        layer_end:       exclusive end index
        is_last_stage:   if True, also include final layer norm + LM head
        device:          string like "cuda:0", "cuda:1", ...
        """
        super().__init__()
        self.device = torch.device(device)
        # 2.1 If this is the very first stage, we need the embedding layer.
        if layer_start == 0:
            self.embed = full_llama.model.embed_tokens.to(self.device)
        else:
            self.embed = None

        # 2.2 Always grab the transformer blocks for [layer_start:layer_end]
        self.layers = nn.ModuleList(
            [blk.to(self.device) for blk in full_llama.model.layers[layer_start:layer_end]]
        )

        # 2.3 If this is the last stage, add final norm + LM head
        self.is_last = is_last_stage
        if is_last_stage:
            self.final_norm = full_llama.model.norm.to(self.device)
            self.lm_head = full_llama.lm_head.to(self.device)
        else:
            self.final_norm = None
            self.lm_head = None

    def forward(self, hidden_states, attention_mask, labels=None):
        """
        If this is Stage 0, `hidden_states` and `attention_mask` will both be None,
        and instead we expect to get `input_ids, attention_mask` directly.
        We handle that special-case in main() rather than here.
        """
        # hidden_states: torch.Tensor [batch, seq_len, hidden_size] on self.device
        # attention_mask: [batch, seq_len] on self.device

        h = hidden_states
        for block in self.layers:
            # Each block returns (output, _) for Llama
            h = block(h, attention_mask=attention_mask)[0]

        if self.is_last:
            h = self.final_norm(h)                 # final layer norm
            logits = self.lm_head(h)               # [batch, seq_len, vocab_size]
            if labels is not None:
                # Shift logits and labels for causal language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                return loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            return logits

        return h, attention_mask  # for non-final stages


# ============================================
# 3) Main: Parses Args, Builds Pipeline, Runs Training Loop
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Pipeline Parallel Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of GPUs to use (i.e. number of pipeline stages)')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate (0.0–1.0)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B', help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Global batch size')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision (FP16)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', '-d', type=str, default='winogrande', help='Dataset to use')
    parser.add_argument('--max_samples', type=int, default=0, help='Maximum number of training samples (0=all)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--eval_steps', type=int, default=50, help='(Optional) steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every N global steps')
    parser.add_argument('--logging_steps', type=int, default=10, help='Log to WandB every N global steps')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--run_id', type=str, required=True, help='Unique run identifier')
    parser.add_argument('-nunf', '--num_unfrozen_layers', type=int, default=None,
                        help='Number of unfrozen layers (if None, all layers are unfrozen)')
    args = parser.parse_args()

    # ──────────────────────────────────────────────────────
    # 3.1 Set Seeds
    # ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # ──────────────────────────────────────────────────────
    # 3.2 Load dataset_config.yaml (unchanged)
    # ──────────────────────────────────────────────────────
    with open("src/src_PP/dataset_config.yaml") as config_file:
        dataset_config_full = yaml.safe_load(config_file)
    dataset_config = dataset_config_full[args.dataset]

    # ──────────────────────────────────────────────────────
    # 3.3 Initialize LossyNetwork (unchanged)
    # ──────────────────────────────────────────────────────
    network = LossyNetwork(args)
    network.set_seed(args.seed)

    # ──────────────────────────────────────────────────────
    # 3.4 Load your classification model + tokenizer (unchanged)
    # ──────────────────────────────────────────────────────
    model_cls, tokenizer = get_classifier_and_tokenizer(
        args.model_name,
        num_labels=dataset_config["num_labels"],
        num_unfrozen_layers=args.num_unfrozen_layers
    )

    # ──────────────────────────────────────────────────────
    # 3.5 Load train & eval datasets (unchanged)
    # ──────────────────────────────────────────────────────
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

    # ──────────────────────────────────────────────────────
    # 3.6 Prepare output directory, save args.yaml (unchanged)
    # ──────────────────────────────────────────────────────
    OUTPUT_DIR_LOCAL = f"{args.output_dir}/{args.run_id}"
    os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)
    with open(f"{OUTPUT_DIR_LOCAL}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # ──────────────────────────────────────────────────────
    # 3.7 Initialize WandB (unchanged)
    # ──────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────
    # 3.8 Pipeline Hyperparameters (from args)
    # ──────────────────────────────────────────────────────
    NUM_GPUS = args.num_nodes
    TOTAL_LAYERS = 32  # Llama 3.2-1B always has 32 transformer blocks
    assert TOTAL_LAYERS % NUM_GPUS == 0, "Total layers (32) must be divisible by --num_nodes"
    LAYERS_PER_STAGE = TOTAL_LAYERS // NUM_GPUS

    GLOBAL_BATCH_SIZE = args.batch_size
    MICRO_BATCHES = NUM_GPUS  # a common heuristic: one micro-batch per GPU (but you could adjust)
    assert GLOBAL_BATCH_SIZE % MICRO_BATCHES == 0, "Global batch size must be divisible by --num_nodes"
    MICRO_BATCH_SIZE = GLOBAL_BATCH_SIZE // MICRO_BATCHES

    LOSSY_FRACTION = args.loss_rate
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    SAVE_EVERY_N_STEPS = args.save_steps
    LOG_EVERY_N_STEPS = args.logging_steps
    MAX_SEQ_LEN = args.max_length

    DEVICE_LIST = [f"cuda:{i}" for i in range(NUM_GPUS)]
    for d in DEVICE_LIST:
        assert torch.cuda.is_available(), f"{d} is not available"

    # ──────────────────────────────────────────────────────
    # 3.9 Load Llama 3.2-1B on CPU & build pipeline stages
    # ──────────────────────────────────────────────────────
    llama_name = args.model_name  # "meta-llama/Llama-3.2-1B"
    full_llama = LlamaForCausalLM.from_pretrained(llama_name).eval()  # load on CPU
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_name)
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

    # Build one stage per GPU:
    stages = []
    for i in range(NUM_GPUS):
        start_idx = i * LAYERS_PER_STAGE
        end_idx = start_idx + LAYERS_PER_STAGE
        is_last = (i == NUM_GPUS - 1)
        stage = PipelineStage(
            full_llama,
            layer_start=start_idx,
            layer_end=end_idx,
            is_last_stage=is_last,
            device=DEVICE_LIST[i]
        )
        stages.append(stage)

    # ──────────────────────────────────────────────────────
    # 3.10 Create an optimizer (one per stage)
    # ──────────────────────────────────────────────────────
    optimizers = []
    for i, stage in enumerate(stages):
        opt = optim.AdamW(stage.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        optimizers.append(opt)

    # ──────────────────────────────────────────────────────
    # 3.11 DataLoader (no DistributedSampler / DDP)
    # ──────────────────────────────────────────────────────
    def collate_fn(batch):
        # Each item in batch: dict {input_ids: [...], attention_mask: [...], labels: int}
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

    # ──────────────────────────────────────────────────────
    # 3.12 (Optional) Eval DataLoader
    # ──────────────────────────────────────────────────────
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # ──────────────────────────────────────────────────────
    # 3.13 If FP16, build a single GradScaler
    # ──────────────────────────────────────────────────────
    scaler = GradScaler() if args.fp16 else None

    # ──────────────────────────────────────────────────────
    # 3.14 Create Callback Instance (for TTAC reporting)
    # ──────────────────────────────────────────────────────
    callback_args = {
        "report_ttac": dataset_config["report_ttac"],
        "report_file": f"{OUTPUT_DIR_LOCAL}/ttac_report.txt",
        "target_acc": dataset_config["target_acc"],
    }
    callback = MyClassifierCallback(callback_args)

    # ──────────────────────────────────────────────────────
    # 3.15 Training Loop: “2*K – 1” Pipeline Schedule
    # ──────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} starting...")
        for batch in train_loader:
            # (a) Split the global batch into MICRO_BATCHES micro-batches
            input_chunks = batch["input_ids"].chunk(MICRO_BATCHES)
            mask_chunks  = batch["attention_mask"].chunk(MICRO_BATCHES)
            label_chunks = batch["labels"].chunk(MICRO_BATCHES)

            # Buffers to hold activations & masks at each pipeline stage
            buf_h = [ [None]*MICRO_BATCHES for _ in range(NUM_GPUS) ]
            buf_m = [ [None]*MICRO_BATCHES for _ in range(NUM_GPUS) ]

            # Buffers to hold losses at the final stage, and gradients flowing upstream
            buf_loss = [None] * MICRO_BATCHES
            grad_buf = [ [None]*MICRO_BATCHES for _ in range(NUM_GPUS-1) ]
            # grad_buf[i][j] will hold gradient from stage (i+1) → stage i for micro-batch j

            # Zero out all optimizers
            for opt in optimizers:
                opt.zero_grad()

            total_steps = 2 * MICRO_BATCHES - 1
            for t in range(total_steps):
                # ───────────────────────────────────────────────────
                # (1) Forward through Stage 0: t < MICRO_BATCHES
                # ───────────────────────────────────────────────────
                if t < MICRO_BATCHES:
                    i = t
                    x0 = input_chunks[i].to(DEVICE_LIST[0])
                    m0 = mask_chunks[i].to(DEVICE_LIST[0])
                    if args.fp16:
                        with autocast():
                            if NUM_GPUS == 1:
                                # If only one stage, Stage0 is also last stage
                                loss0 = stages[0](x0, m0, labels=label_chunks[i].to(DEVICE_LIST[0]))
                                buf_loss[i] = loss0
                            else:
                                h0, a0 = stages[0](x0, m0)
                                buf_h[0][i] = h0.detach()
                                buf_m[0][i] = a0.detach()
                    else:
                        if NUM_GPUS == 1:
                            loss0 = stages[0](x0, m0, labels=label_chunks[i].to(DEVICE_LIST[0]))
                            buf_loss[i] = loss0
                        else:
                            h0, a0 = stages[0](x0, m0)
                            buf_h[0][i] = h0.detach()
                            buf_m[0][i] = a0.detach()

                # ───────────────────────────────────────────────────
                # (2) Forward through intermediate stages 1..(N-2)
                #     Stage k consumes buf_h[k-1][i] when  (t >= k) and (t <= MICRO_BATCHES + k - 1)
                # ───────────────────────────────────────────────────
                for k in range(1, NUM_GPUS-1):
                    if k <= t < (MICRO_BATCHES + k):
                        i = t - k
                        h_in = buf_h[k-1][i].to(DEVICE_LIST[k]).requires_grad_(True)
                        m_in = buf_m[k-1][i].to(DEVICE_LIST[k])
                        if args.fp16:
                            with autocast():
                                h_out, a_out = stages[k](h_in, m_in)
                        else:
                            h_out, a_out = stages[k](h_in, m_in)
                        buf_h[k][i] = h_out.detach()
                        buf_m[k][i] = a_out.detach()

                # ───────────────────────────────────────────────────
                # (3) Forward + Loss through Final Stage (k = N-1)
                #     Occurs when t in [ (N-1) , (MICRO_BATCHES + N-2 ) ]
                # ───────────────────────────────────────────────────
                last_stage = NUM_GPUS - 1
                if (NUM_GPUS - 1) <= t < (MICRO_BATCHES + NUM_GPUS - 1):
                    i = t - (NUM_GPUS - 1)
                    # Input to final stage is buf_h[last_stage-1][i]
                    h_in = buf_h[last_stage-1][i].to(DEVICE_LIST[last_stage]).requires_grad_(True)
                    m_in = buf_m[last_stage-1][i].to(DEVICE_LIST[last_stage])
                    label_i = label_chunks[i].to(DEVICE_LIST[last_stage])
                    if args.fp16:
                        with autocast():
                            loss_i = stages[last_stage](h_in, m_in, labels=label_i)
                        buf_loss[i] = loss_i
                    else:
                        loss_i = stages[last_stage](h_in, m_in, labels=label_i)
                        buf_loss[i] = loss_i

                # ───────────────────────────────────────────────────
                # (4) Backward through Final Stage: t ≥ (MICRO_BATCHES + N-1)
                #     Loss backpropagates to obtain ∂L/∂h_in on device last_stage
                # ───────────────────────────────────────────────────
                if t >= (MICRO_BATCHES + NUM_GPUS - 1):
                    i = t - (MICRO_BATCHES + NUM_GPUS - 1)
                    if args.fp16:
                        scaler.scale(buf_loss[i]).backward(retain_graph=True)
                    else:
                        buf_loss[i].backward(retain_graph=True)
                    # Now h_in.grad holds ∂L/∂h_in for the final stage
                    g = h_in.grad                           # on cuda:(N-1)
                    gz = zero_some_fraction(g, LOSSY_FRACTION)
                    grad_buf[last_stage-1][i] = gz.to(DEVICE_LIST[last_stage-1])

                # ───────────────────────────────────────────────────
                # (5) Backward through intermediate stages (k from N-2 down to 1)
                #     At step t, for each k s.t. t ≥ (MICRO_BATCHES + k)
                # ───────────────────────────────────────────────────
                for k in range(NUM_GPUS - 2, 0, -1):
                    if t >= (MICRO_BATCHES + k):
                        i = t - (MICRO_BATCHES + k)
                        # Recompute forward for stage k:
                        h_in_rec = buf_h[k-1][i].to(DEVICE_LIST[k]).requires_grad_(True)
                        m_in_rec = buf_m[k-1][i].to(DEVICE_LIST[k])
                        if args.fp16:
                            with autocast():
                                h_out_rec, _ = stages[k](h_in_rec, m_in_rec)
                        else:
                            h_out_rec, _ = stages[k](h_in_rec, m_in_rec)
                        # Backprop through stage k using gradient from k+1:
                        h_out_rec.backward(grad_buf[k][i], retain_graph=True)
                        g = h_in_rec.grad                # ∂L/∂h_in on cuda:k
                        gz = zero_some_fraction(g, LOSSY_FRACTION)
                        grad_buf[k-1][i] = gz.to(DEVICE_LIST[k-1])

                # ───────────────────────────────────────────────────
                # (6) Backward through Stage 0: t ≥ (MICRO_BATCHES + 0)
                # ───────────────────────────────────────────────────
                if t >= MICRO_BATCHES:
                    i = t - MICRO_BATCHES
                    # Recompute forward for stage0:
                    x0_rec = input_chunks[i].to(DEVICE_LIST[0]).requires_grad_(True)
                    m0_rec = mask_chunks[i].to(DEVICE_LIST[0])
                    if args.fp16:
                        with autocast():
                            if NUM_GPUS == 1:
                                # Single-stage FP16
                                loss0_rec = stages[0](x0_rec, m0_rec, labels=label_chunks[i].to(DEVICE_LIST[0]))
                                scaler.scale(loss0_rec).backward(retain_graph=True)
                            else:
                                h0_rec, _ = stages[0](x0_rec, m0_rec)
                                h0_rec.backward(grad_buf[0][i], retain_graph=True)
                    else:
                        if NUM_GPUS == 1:
                            loss0_rec = stages[0](x0_rec, m0_rec, labels=label_chunks[i].to(DEVICE_LIST[0]))
                            loss0_rec.backward(retain_graph=True)
                        else:
                            h0_rec, _ = stages[0](x0_rec, m0_rec)
                            h0_rec.backward(grad_buf[0][i], retain_graph=True)

            # ───────────────────────────────────────────────────
            # (b) Optimizer Step: update each stage’s parameters
            # ───────────────────────────────────────────────────
            if args.fp16:
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in optimizers:
                    opt.step()

            # ───────────────────────────────────────────────────
            # (c) Logging & Checkpointing
            # ───────────────────────────────────────────────────
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
                # Save each stage and optimizer state
                for i in range(NUM_GPUS):
                    ckpt[f"stage{i}_state"] = stages[i].state_dict()
                    ckpt[f"optimizer{i}_state"] = optimizers[i].state_dict()
                ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, f"checkpoint-step{global_step}.pt")
                torch.save(ckpt, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        # End of epoch
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} complete.")

        # ───────────────────────────────────────────────────
        # (Optional) Run Evaluation & Call Callbacks if desired
        # You can run an evaluation pass over eval_loader here, gather
        # logits/labels, compute metrics using `compute_classfication_metrics`,
        # and trigger `MyClassifierCallback.on_evaluate(...)` exactly as before.
        # (Not shown in full for brevity.)
        # ───────────────────────────────────────────────────

    print("Training finished!")
    wandb.finish()


if __name__ == "__main__":
    main()
