# # main.py

# import os
# import yaml
# import random
# import torch
# import argparse

# from comms import LossyNetwork
# from data import get_dataset
# from models import get_classifier_and_tokenizer
# from trainer import MyClassifierCallback, compute_classfication_metrics

# import wandb
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler


# # ============================================
# # 1) Helper: Zero out a fraction of a Tensor
# # ============================================
# def zero_some_fraction(tensor: torch.Tensor, percent: float) -> torch.Tensor:
#     """
#     Randomly zero out `percent` fraction of elements in `tensor`.
#     Returns a new tensor (same device & dtype) with that fraction zeroed.
#     """
#     if percent <= 0.0:
#         return tensor
#     mask = (torch.rand_like(tensor) >= percent).to(tensor.dtype)
#     return tensor * mask


# # ============================================
# # 2.A) Stage0: Embedding + transformer blocks
# # ============================================
# class Stage0(nn.Module):
#     """
#     Stage 0: holds the embedding layer + a contiguous slice of transformer blocks.
#     Expects input: (input_ids: LongTensor, attention_mask: LongTensor) on cuda:0
#     Returns: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:0
#     """
#     def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
#         """
#         full_model:  AutoModelForSequenceClassification loaded on CPU
#         layer_start/layer_end: indices into full_model.model.layers
#         device:      e.g. "cuda:0"
#         """
#         super().__init__()
#         self.device = torch.device(device)

#         # Embedding from the base model
#         self.embed = full_model.model.embed_tokens.to(self.device)

#         # A slice of transformer blocks [layer_start:layer_end]
#         self.layers = nn.ModuleList(
#             [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
#         )

#     def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
#         """
#         input_ids:      LongTensor [batch, seq_len] on cuda:0
#         attention_mask: LongTensor [batch, seq_len] on cuda:0
#         """
#         # (1) Embed tokens → [batch, seq_len, hidden_dim]
#         h = self.embed(input_ids)  # -> (batch, seq_len, hidden_dim)

#         # (2) Build position IDs: 0..(seq_len-1) for every batch
#         batch_size, seq_len = input_ids.size()
#         position_ids = (
#             torch.arange(seq_len, device=self.device)
#             .unsqueeze(0)
#             .expand(batch_size, seq_len)
#         )  # [batch, seq_len] long

#         # (3) Pass through each transformer block, supplying position_ids
#         for block in self.layers:
#             h = block(
#                 h,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids
#             )[0]

#         # (4) Return hidden states + attention mask for the next stage
#         return h, attention_mask


# # ============================================
# # 2.B) StageMiddle: Intermediate transformer blocks
# # ============================================
# class StageMiddle(nn.Module):
#     """
#     Stage k where 1 ≤ k ≤ N-2.  Holds a contiguous slice of transformer blocks.
#     Expects input: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:k
#     Returns: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:k
#     """
#     def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
#         super().__init__()
#         self.device = torch.device(device)
#         self.layers = nn.ModuleList(
#             [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
#         )

#     def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor):
#         """
#         hidden_states:  FloatTensor [batch, seq_len, hidden_dim] on cuda:k
#         attention_mask: LongTensor [batch, seq_len] on cuda:k
#         """
#         h = hidden_states  # [batch, seq_len, hidden_dim]
#         batch_size, seq_len, _ = h.size()

#         # (1) Build position IDs
#         position_ids = (
#             torch.arange(seq_len, device=self.device)
#             .unsqueeze(0)
#             .expand(batch_size, seq_len)
#         )  # [batch, seq_len] long

#         # (2) Pass through each transformer block, supplying position_ids
#         for block in self.layers:
#             h = block(
#                 h,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids
#             )[0]

#         return h, attention_mask


# # ============================================
# # 2.C) StageLast: Final transformer blocks + norm + classification head
# # ============================================
# class StageLast(nn.Module):
#     """
#     Stage N-1: holds a contiguous slice of transformer blocks,
#     plus final LayerNorm and classification head.
#     Expects input: (hidden_states: FloatTensor, attention_mask: LongTensor) on cuda:N-1
#     If labels are given, returns scalar loss; otherwise returns logits.
#     """
#     def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
#         super().__init__()
#         self.device = torch.device(device)
#         self.layers = nn.ModuleList(
#             [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
#         )
#         self.final_norm = full_model.model.norm.to(self.device)
#         self.classifier = full_model.score.to(self.device)

#     def forward(self, hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor, labels=None):
#         """
#         hidden_states:  FloatTensor [batch, seq_len, hidden_dim] on cuda:N-1
#         attention_mask: LongTensor [batch, seq_len] on cuda:N-1
#         labels:         LongTensor [batch] on cuda:N-1 (optional)
#         """
#         h = hidden_states  # [batch, seq_len, hidden_dim]
#         batch_size, seq_len, _ = h.size()

#         # (1) Build position IDs
#         position_ids = (
#             torch.arange(seq_len, device=self.device)
#             .unsqueeze(0)
#             .expand(batch_size, seq_len)
#         )  # [batch, seq_len] long

#         # (2) Pass through each transformer block, supplying position_ids
#         for block in self.layers:
#             h = block(
#                 h,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids
#             )[0]

#         # (3) Final norm + classification head
#         h_norm = self.final_norm(h)                 # [batch, seq_len, hidden_dim]
#         cls_token = h_norm[:, 0, :]                 # [batch, hidden_dim]
#         logits = self.classifier(cls_token)         # [batch, num_labels]

#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             return loss_fct(logits, labels)

#         return logits


# # ============================================
# # 3) Main Training Function
# # ============================================
# def main():
#     parser = argparse.ArgumentParser(description="Pipeline Parallel Classification with Packet Loss")
#     parser.add_argument('--num_nodes', type=int, default=2,
#                         help='Number of GPUs to use (i.e. number of pipeline stages)')
#     parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate (0.0–1.0)')
#     parser.add_argument('--seed', type=int, default=1234, help='Random seed')
#     parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B',
#                         help='Model name (checkpoint for AutoModelForSequenceClassification)')
#     parser.add_argument('--batch_size', type=int, default=64, help='Global batch size (must divide by num_nodes)')
#     parser.add_argument('--fp16', action='store_true', help='Use mixed precision (FP16)')
#     parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
#     parser.add_argument('--dataset', '-d', type=str, default='winogrande', help='Dataset to use')
#     parser.add_argument('--max_samples', type=int, default=0,
#                         help='Maximum number of training samples to use (0 = all)')
#     parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
#     parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
#     parser.add_argument('--eval_steps', type=int, default=50, help='(Optional) steps between evaluations')
#     parser.add_argument('--save_steps', type=int, default=100, help='Save checkpoint every N global steps')
#     parser.add_argument('--logging_steps', type=int, default=10, help='Log to WandB every N global steps')
#     parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
#     parser.add_argument('--run_id', type=str, required=True, help='Unique run identifier')
#     parser.add_argument('-nunf', '--num_unfrozen_layers', type=int, default=None,
#                         help='Number of last transformer layers to unfreeze (others are frozen)')
#     args = parser.parse_args()

#     # ──────────────────────────────────────────────────────────
#     # 3.1 Set Random Seeds
#     # ──────────────────────────────────────────────────────────
#     torch.manual_seed(args.seed)
#     random.seed(args.seed)

#     # ──────────────────────────────────────────────────────────
#     # 3.2 Load dataset_config.yaml
#     # ──────────────────────────────────────────────────────────
#     with open("src/src_PP/dataset_config.yaml") as config_file:
#         dataset_config_full = yaml.safe_load(config_file)
#     dataset_config = dataset_config_full[args.dataset]

#     # ──────────────────────────────────────────────────────────
#     # 3.3 Initialize LossyNetwork
#     # ──────────────────────────────────────────────────────────
#     network = LossyNetwork(args)
#     network.set_seed(args.seed)

#     # ──────────────────────────────────────────────────────────
#     # 3.4 Load classification model + tokenizer via models.py
#     # ──────────────────────────────────────────────────────────
#     model_cls, tokenizer = get_classifier_and_tokenizer(
#         args.model_name,
#         num_labels=dataset_config["num_labels"],
#         num_unfrozen_layers=args.num_unfrozen_layers
#     )
#     # `model_cls` is AutoModelForSequenceClassification (e.g. LlamaForSequenceClassification),
#     # possibly with early layers frozen if `nunf` is specified.

#     # ──────────────────────────────────────────────────────────
#     # 3.5 Load train & eval datasets
#     # ──────────────────────────────────────────────────────────
#     train_dataset, eval_dataset = get_dataset(args, tokenizer)

#     # ──────────────────────────────────────────────────────────
#     # 3.6 Prepare output directory, save args.yaml
#     # ──────────────────────────────────────────────────────────
#     OUTPUT_DIR_LOCAL = f"{args.output_dir}/{args.run_id}"
#     os.makedirs(OUTPUT_DIR_LOCAL, exist_ok=True)
#     with open(f"{OUTPUT_DIR_LOCAL}/args.yaml", "w") as f:
#         yaml.dump(vars(args), f)

#     # ──────────────────────────────────────────────────────────
#     # 3.7 Initialize WandB
#     # ──────────────────────────────────────────────────────────
#     wandb.init(
#         project="your_project_name",
#         name=args.run_id,
#         reinit=True
#     )
#     wandb.config.update({
#         "dataset": args.dataset,
#         "run_id": args.run_id,
#         "epochs": args.epochs,
#         "batch_size": args.batch_size,
#         "loss_rate": args.loss_rate,
#         "learning_rate": args.learning_rate,
#         "num_unfrozen_layers": args.num_unfrozen_layers,
#         "num_nodes": args.num_nodes,
#         "fp16": args.fp16,
#     })

#     # ──────────────────────────────────────────────────────────
#     # 3.8 Pipeline Hyperparameters (from args)
#     # ──────────────────────────────────────────────────────────
#     NUM_GPUS = args.num_nodes
#     TOTAL_LAYERS = model_cls.model.config.num_hidden_layers  # e.g. 32 for Llama-3.2-1B
#     assert TOTAL_LAYERS % NUM_GPUS == 0, f"Total layers ({TOTAL_LAYERS}) must be divisible by num_nodes"
#     LAYERS_PER_STAGE = TOTAL_LAYERS // NUM_GPUS

#     GLOBAL_BATCH_SIZE = args.batch_size
#     MICRO_BATCHES = NUM_GPUS   # Often one micro-batch per GPU
#     assert GLOBAL_BATCH_SIZE % MICRO_BATCHES == 0, "Global batch size must be divisible by num_nodes"
#     MICRO_BATCH_SIZE = GLOBAL_BATCH_SIZE // MICRO_BATCHES

#     LOSSY_FRACTION = args.loss_rate
#     NUM_EPOCHS = args.epochs
#     LEARNING_RATE = args.learning_rate
#     SAVE_EVERY_N_STEPS = args.save_steps
#     LOG_EVERY_N_STEPS = args.logging_steps
#     MAX_SEQ_LEN = args.max_length

#     DEVICE_LIST = [f"cuda:{i}" for i in range(NUM_GPUS)]
#     for dev in DEVICE_LIST:
#         if not torch.cuda.is_available():
#             raise RuntimeError(f"{dev} is not available")

#     # ──────────────────────────────────────────────────────────
#     # 3.9 Handle Single‐GPU (NUM_GPUS == 1)
#     # ──────────────────────────────────────────────────────────
#     if NUM_GPUS == 1:
#         # Standard single‐GPU fine‐tuning with model_cls
#         device = torch.device("cuda:0")
#         model_cls.to(device)
#         optimizer = optim.AdamW(model_cls.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
#         scaler = GradScaler() if args.fp16 else None

#         # DataLoader (shared collate_fn)
#         def collate_fn(batch):
#             input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
#             attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
#             labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
#             return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=GLOBAL_BATCH_SIZE,
#             shuffle=True,
#             collate_fn=collate_fn,
#             num_workers=4,
#             pin_memory=True,
#         )

#         callback_args = {
#             "report_ttac": dataset_config["report_ttac"],
#             "report_file": f"{OUTPUT_DIR_LOCAL}/ttac_report.txt",
#             "target_acc": dataset_config["target_acc"],
#         }
#         callback = MyClassifierCallback(callback_args)

#         global_step = 0
#         for epoch in range(NUM_EPOCHS):
#             print(f"[Single‐GPU] Epoch {epoch+1}/{NUM_EPOCHS} starting...")
#             for batch in train_loader:
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["labels"].to(device)

#                 optimizer.zero_grad()
#                 if args.fp16:
#                     with autocast():
#                         outputs = model_cls(
#                             input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             labels=labels
#                         )
#                         loss = outputs.loss
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
#                 else:
#                     outputs = model_cls(
#                         input_ids=input_ids,
#                         attention_mask=attention_mask,
#                         labels=labels
#                     )
#                     loss = outputs.loss
#                     loss.backward()
#                     optimizer.step()

#                 global_step += 1
#                 if global_step % LOG_EVERY_N_STEPS == 0:
#                     wandb.log({
#                         "epoch": epoch,
#                         "global_step": global_step,
#                         "train_loss": loss.item()
#                     })

#                 if global_step % SAVE_EVERY_N_STEPS == 0:
#                     ckpt = {
#                         "global_step": global_step,
#                         "epoch": epoch,
#                         "model_state": model_cls.state_dict(),
#                         "optimizer_state": optimizer.state_dict(),
#                     }
#                     ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, f"checkpoint-step{global_step}.pt")
#                     torch.save(ckpt, ckpt_path)
#                     print(f"Saved checkpoint to {ckpt_path}")

#             print(f"[Single‐GPU] Epoch {epoch+1}/{NUM_EPOCHS} complete.")
#         print("Single‐GPU training finished!")
#         wandb.finish()
#         return

#     # ──────────────────────────────────────────────────────────
#     # 3.10 Instantiate Each Pipeline Stage (NUM_GPUS ≥ 2)
#     # ──────────────────────────────────────────────────────────
#     stages = []
#     for i in range(NUM_GPUS):
#         start_idx = i * LAYERS_PER_STAGE
#         end_idx = start_idx + LAYERS_PER_STAGE
#         is_last = (i == NUM_GPUS - 1)
#         if i == 0:
#             stage = Stage0(
#                 full_model=model_cls,
#                 layer_start=start_idx,
#                 layer_end=end_idx,
#                 device=DEVICE_LIST[i]
#             )
#         elif is_last:
#             stage = StageLast(
#                 full_model=model_cls,
#                 layer_start=start_idx,
#                 layer_end=end_idx,
#                 device=DEVICE_LIST[i]
#             )
#         else:
#             stage = StageMiddle(
#                 full_model=model_cls,
#                 layer_start=start_idx,
#                 layer_end=end_idx,
#                 device=DEVICE_LIST[i]
#             )
#         stages.append(stage)

#     # ──────────────────────────────────────────────────────────
#     # 3.11 Create One Optimizer per Stage
#     # ──────────────────────────────────────────────────────────
#     optimizers = []
#     for st in stages:
#         opt = optim.AdamW(st.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
#         optimizers.append(opt)

#     # ──────────────────────────────────────────────────────────
#     # 3.12 Prepare DataLoader (no DistributedSampler)
#     # ──────────────────────────────────────────────────────────
#     def collate_fn(batch):
#         input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
#         attention_mask = torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long)
#         labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
#         return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=GLOBAL_BATCH_SIZE,
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True,
#     )

#     eval_loader = DataLoader(
#         eval_dataset,
#         batch_size=GLOBAL_BATCH_SIZE,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True,
#     )

#     # ──────────────────────────────────────────────────────────
#     # 3.13 Setup Mixed Precision if Requested
#     # ──────────────────────────────────────────────────────────
#     scaler = GradScaler() if args.fp16 else None

#     # ──────────────────────────────────────────────────────────
#     # 3.14 Create TTAC Callback (unchanged)
#     # ──────────────────────────────────────────────────────────
#     callback_args = {
#         "report_ttac": dataset_config["report_ttac"],
#         "report_file": f"{OUTPUT_DIR_LOCAL}/ttac_report.txt",
#         "target_acc": dataset_config["target_acc"],
#     }
#     callback = MyClassifierCallback(callback_args)

#     # ──────────────────────────────────────────────────────────
#     # 3.15 Training Loop: 2*K – 1 Pipeline Schedule
#     # ──────────────────────────────────────────────────────────
#     global_step = 0
#     for epoch in range(NUM_EPOCHS):
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS} starting...")
#         for batch in train_loader:
#             # (a) Split the global batch into MICRO_BATCHES micro-batches
#             input_chunks = batch["input_ids"].chunk(MICRO_BATCHES)
#             mask_chunks = batch["attention_mask"].chunk(MICRO_BATCHES)
#             label_chunks = batch["labels"].chunk(MICRO_BATCHES)

#             # Buffers for activations & masks at each stage
#             buf_h = [[None] * MICRO_BATCHES for _ in range(NUM_GPUS)]
#             buf_m = [[None] * MICRO_BATCHES for _ in range(NUM_GPUS)]

#             # Buffers for final losses and upstream gradients
#             buf_loss = [None] * MICRO_BATCHES
#             grad_buf = [[None] * MICRO_BATCHES for _ in range(NUM_GPUS - 1)]

#             # Zero all optimizers
#             for opt in optimizers:
#                 opt.zero_grad()

#             total_steps = 2 * MICRO_BATCHES - 1
#             for t in range(total_steps):
#                 # ───────────────────────────────────────────────────
#                 # (1) Stage 0 Forward: t < MICRO_BATCHES
#                 # ───────────────────────────────────────────────────
#                 if t < MICRO_BATCHES:
#                     i = t
#                     x0 = input_chunks[i].to(DEVICE_LIST[0])    # [batch, seq_len] LongTensor
#                     m0 = mask_chunks[i].to(DEVICE_LIST[0])     # [batch, seq_len] LongTensor

#                     if args.fp16:
#                         with autocast():
#                             h0, a0 = stages[0](x0, m0)
#                     else:
#                         h0, a0 = stages[0](x0, m0)

#                     buf_h[0][i] = h0.detach()
#                     buf_m[0][i] = a0.detach()

#                 # ───────────────────────────────────────────────────
#                 # (2) Stage k Forward for 1 ≤ k ≤ N-2:
#                 #     Condition: k ≤ t < MICRO_BATCHES + k
#                 # ───────────────────────────────────────────────────
#                 for k in range(1, NUM_GPUS - 1):
#                     if k <= t < (MICRO_BATCHES + k):
#                         i = t - k
#                         h_in = buf_h[k - 1][i].to(DEVICE_LIST[k]).requires_grad_(True)
#                         m_in = buf_m[k - 1][i].to(DEVICE_LIST[k])
#                         if args.fp16:
#                             with autocast():
#                                 h_out, a_out = stages[k](h_in, m_in)
#                         else:
#                             h_out, a_out = stages[k](h_in, m_in)
#                         buf_h[k][i] = h_out.detach()
#                         buf_m[k][i] = a_out.detach()

#                 # ───────────────────────────────────────────────────
#                 # (3) Stage N-1 (Last) Forward + Loss:
#                 #     Condition: (N-1) ≤ t < MICRO_BATCHES + (N-1)
#                 # ───────────────────────────────────────────────────
#                 last_stage = NUM_GPUS - 1
#                 if (NUM_GPUS - 1) <= t < (MICRO_BATCHES + NUM_GPUS - 1):
#                     i = t - (NUM_GPUS - 1)
#                     h_last_in = buf_h[last_stage - 1][i].to(DEVICE_LIST[last_stage]).requires_grad_(True)
#                     m_last_in = buf_m[last_stage - 1][i].to(DEVICE_LIST[last_stage])
#                     lbl = label_chunks[i].to(DEVICE_LIST[last_stage])

#                     if args.fp16:
#                         with autocast():
#                             loss_i = stages[last_stage](h_last_in, m_last_in, labels=lbl)
#                         buf_loss[i] = loss_i
#                     else:
#                         loss_i = stages[last_stage](h_last_in, m_last_in, labels=lbl)
#                         buf_loss[i] = loss_i

#                 # ───────────────────────────────────────────────────
#                 # (4) Stage N-1 Backward: t ≥ (MICRO_BATCHES + (N-1))
#                 # ───────────────────────────────────────────────────
#                 if t >= (MICRO_BATCHES + NUM_GPUS - 1):
#                     i = t - (MICRO_BATCHES + NUM_GPUS - 1)
#                     if args.fp16:
#                         scaler.scale(buf_loss[i]).backward(retain_graph=True)
#                     else:
#                         buf_loss[i].backward(retain_graph=True)
#                     # Capture gradient wrt h_last_in on cuda:(N-1)
#                     g_last = h_last_in.grad
#                     gz = zero_some_fraction(g_last, LOSSY_FRACTION)
#                     grad_buf[last_stage - 1][i] = gz.to(DEVICE_LIST[last_stage - 1])

#                 # ───────────────────────────────────────────────────
#                 # (5) Backward through Stages N-2 down to 1:
#                 #     For each k: if t ≥ (MICRO_BATCHES + k)
#                 # ───────────────────────────────────────────────────
#                 for k in range(NUM_GPUS - 2, 0, -1):
#                     if t >= (MICRO_BATCHES + k):
#                         i = t - (MICRO_BATCHES + k)
#                         # Recompute forward for stage k
#                         h_in_rec = buf_h[k - 1][i].to(DEVICE_LIST[k]).requires_grad_(True)
#                         m_in_rec = buf_m[k - 1][i].to(DEVICE_LIST[k])
#                         if args.fp16:
#                             with autocast():
#                                 h_out_rec, _ = stages[k](h_in_rec, m_in_rec)
#                         else:
#                             h_out_rec, _ = stages[k](h_in_rec, m_in_rec)
#                         # Backprop through stage k
#                         h_out_rec.backward(grad_buf[k][i], retain_graph=True)
#                         gk = h_in_rec.grad
#                         gkz = zero_some_fraction(gk, LOSSY_FRACTION)
#                         grad_buf[k - 1][i] = gkz.to(DEVICE_LIST[k - 1])

#                 # ───────────────────────────────────────────────────
#                 # (6) Backward through Stage 0: t ≥ MICRO_BATCHES
#                 # ───────────────────────────────────────────────────
#                 if t >= MICRO_BATCHES:
#                     i = t - MICRO_BATCHES
#                     x0_rec = input_chunks[i].to(DEVICE_LIST[0]).requires_grad_(True)
#                     m0_rec = mask_chunks[i].to(DEVICE_LIST[0])
#                     if args.fp16:
#                         with autocast():
#                             h0_rec, _ = stages[0](x0_rec, m0_rec)
#                             h0_rec.backward(grad_buf[0][i], retain_graph=True)
#                     else:
#                         h0_rec, _ = stages[0](x0_rec, m0_rec)
#                         h0_rec.backward(grad_buf[0][i], retain_graph=True)

#             # ──────────────────────────────────────────────────────────
#             # (b) Optimizer Step: update each stage’s parameters
#             # ──────────────────────────────────────────────────────────
#             if args.fp16:
#                 for opt in optimizers:
#                     scaler.step(opt)
#                 scaler.update()
#             else:
#                 for opt in optimizers:
#                     opt.step()

#             # ──────────────────────────────────────────────────────────
#             # (c) Logging & Checkpointing
#             # ──────────────────────────────────────────────────────────
#             global_step += 1
#             micro_losses = [buf_loss[i].item() for i in range(MICRO_BATCHES)]
#             avg_loss = sum(micro_losses) / MICRO_BATCHES

#             if global_step % LOG_EVERY_N_STEPS == 0:
#                 wandb.log({
#                     "epoch": epoch,
#                     "global_step": global_step,
#                     "train_loss": avg_loss
#                 })

#             if global_step % SAVE_EVERY_N_STEPS == 0:
#                 ckpt = {
#                     "global_step": global_step,
#                     "epoch": epoch,
#                 }
#                 for i in range(NUM_GPUS):
#                     ckpt[f"stage{i}_state"] = stages[i].state_dict()
#                     ckpt[f"optimizer{i}_state"] = optimizers[i].state_dict()
#                 ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, f"checkpoint-step{global_step}.pt")
#                 torch.save(ckpt, ckpt_path)
#                 print(f"Saved checkpoint to {ckpt_path}")

#         # End of epoch
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS} complete.")

#         # ───────────────────────────────────────────────────────
#         # (Optional) Evaluation & Callback invocation here
#         # You can run an evaluation loop over eval_loader, call
#         # compute_classfication_metrics, then invoke callback.on_evaluate(...)
#         # exactly as before.
#         # ───────────────────────────────────────────────────────

#     print("Training finished!")
#     wandb.finish()


# if __name__ == "__main__":
#     main()






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
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

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
    Stage 0: the embedding layer + a slice of transformer blocks.
    Expects: input_ids (LongTensor) and attention_mask (LongTensor), both on cuda:0
    Returns: (hidden_states, attention_mask, position_ids)  all on cuda:0
    """
    def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
        super().__init__()
        self.device = torch.device(device)

        # (a) The embedding layer is the same one inside `full_model.model.embed_tokens`
        self.embed = full_model.model.embed_tokens.to(self.device)

        # # (b) A contiguous slice of transformer blocks from layer_start to layer_end
        # self.layers = nn.ModuleList(
        #     [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
        # )
        
        self.layers = nn.ModuleList()
        for blk in full_model.model.layers[layer_start:layer_end]:
            blk = blk.to(self.device)
            # ──────────────────────────────────────────────────────────────────────
            # Force-reassign rotary_emb on every attention in this block:
            # (so that block.self_attn.rotary_emb is never None on cuda:0)
            blk.self_attn.rotary_emb = LlamaRotaryEmbedding(
                full_model.model.config
            ).to(self.device)
            # ──────────────────────────────────────────────────────────────────────
            self.layers.append(blk)

        # ──────────────────────────────────────────────
        # NOTE: We do NOT re-assign rotary_emb here.  The original
        # LlamaForSequenceClassification initialization already did:
        #    block.self_attn.rotary_emb = LlamaRotaryEmbedding(config).to(block.device)
        # under the hood.  By simply moving `blk.to(self.device)`, we keep that `rotary_emb`.
        # If we had overwritten rotary_emb here, we risk clobbering it.
        # ──────────────────────────────────────────────

# For git, remove later

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        input_ids:      LongTensor of shape [batch, seq_len] on cuda:0
        attention_mask: LongTensor of shape [batch, seq_len] on cuda:0

        Returns:
          - hidden_states: FloatTensor [batch, seq_len, hidden_dim] on cuda:0
          - attention_mask: same attention_mask, passed onward
          - position_ids: LongTensor [batch, seq_len] on cuda:0
        """
        # 1) Embed tokens
        h = self.embed(input_ids)  # [batch, seq_len, hidden_dim]

        # 2) Build position_ids = [0,1,2,..., seq_len-1] for each example in the batch
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, seq_len)
        # shape: [batch, seq_len]

        # 3) Pass through each transformer block, supplying attention_mask & position_ids
        for block in self.layers:
            # block(...) returns a tuple (hidden_states, present_key_value) but we only need [0]
            h = block(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]  # [batch, seq_len, hidden_dim]

        # 4) Return hidden+mask+pos_ids for next stage
        return h, attention_mask, position_ids
  


# ============================================
# 2.B) StageMiddle: Intermediate transformer blocks
# ============================================
class StageMiddle(nn.Module):
    """
    Stage k (1 ≤ k ≤ N-2): holds a contiguous slice of transformer blocks,
    each expecting (hidden_states, attention_mask, position_ids) on cuda:k.
    Returns: (hidden_out, attention_mask, position_ids) on cuda:k
    """
    def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
        super().__init__()
        self.device = torch.device(device)

        # # A contiguous slice of transformer blocks [layer_start:layer_end]
        # self.layers = nn.ModuleList(
        #     [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
        # )
        
        # A contiguous slice of transformer blocks [layer_start:layer_end]
        self.layers = nn.ModuleList()
        for blk in full_model.model.layers[layer_start:layer_end]:
            blk = blk.to(self.device)
            # ───────────────────────────────────────────────────────────────────────────
            # Force-reassign rotary_emb on every attention in this block:
            blk.self_attn.rotary_emb = LlamaRotaryEmbedding(
                full_model.model.config
            ).to(self.device)
            # ───────────────────────────────────────────────────────────────────────────
            self.layers.append(blk)

        # Again—DO NOT reassign rotary_emb here.  The pre-trained blocks already have it.

    def forward(self,
                hidden_states: torch.FloatTensor,
                attention_mask: torch.LongTensor,
                position_ids: torch.LongTensor):
        """
        hidden_states:  FloatTensor [batch, seq_len, hidden_dim] on cuda:k
        attention_mask: LongTensor [batch, seq_len] on cuda:k
        position_ids:   LongTensor [batch, seq_len] on cuda:k

        Returns: (hidden_out, attention_mask, position_ids) on cuda:k
        """
        h = hidden_states
        # We reuse the same position_ids throughout the slice of blocks
        for block in self.layers:
            h = block(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        # Pass both mask + pos_ids unchanged downstream
        return h, attention_mask, position_ids



# ============================================
# 2.C) StageLast: Final transformer blocks + norm + classification head
# ============================================
class StageLast(nn.Module):
    """
    Stage N-1: the final slice of transformer blocks,
    plus the final LayerNorm and classification head.
    Expects: (hidden_states, attention_mask, position_ids[, labels]) on cuda:N-1
    If `labels` is provided → return a scalar cross‐entropy loss.
    Otherwise → return logits [batch, num_labels].
    """
    def __init__(self, full_model, layer_start: int, layer_end: int, device: str):
        super().__init__()
        self.device = torch.device(device)

        # # Slice of transformer blocks [layer_start:layer_end]
        # self.layers = nn.ModuleList(
        #     [blk.to(self.device) for blk in full_model.model.layers[layer_start:layer_end]]
        # )
        
        # Slice of transformer blocks [layer_start:layer_end]
        self.layers = nn.ModuleList()
        for blk in full_model.model.layers[layer_start:layer_end]:
            blk = blk.to(self.device)
            # ───────────────────────────────────────────────────────────────────────────
            # Force-reassign rotary_emb on every attention in this block:
            blk.self_attn.rotary_emb = LlamaRotaryEmbedding(
                full_model.model.config
            ).to(self.device)
            # ───────────────────────────────────────────────────────────────────────────
            self.layers.append(blk)

        # Final LayerNorm
        self.final_norm = full_model.model.norm.to(self.device)

        # Classification head (a linear layer on top of hidden_dim → num_labels)
        self.classifier = full_model.score.to(self.device)

        # Again, do NOT overwrite rotary_emb here.  The pre-trained blocks already have it.

    def forward(self,
                hidden_states: torch.FloatTensor,
                attention_mask: torch.LongTensor,
                position_ids: torch.LongTensor,
                labels: torch.LongTensor = None):
        """
        hidden_states:  FloatTensor [batch, seq_len, hidden_dim] on cuda:N-1
        attention_mask: LongTensor [batch, seq_len] on cuda:N-1
        position_ids:   LongTensor [batch, seq_len] on cuda:N-1
        labels:         LongTensor [batch] on cuda:N-1  (optional)

        Returns:
          - If `labels` is not None: a scalar loss (CrossEntropyLoss).
          - Else: logits [batch, num_labels].
        """
        h = hidden_states
        for block in self.layers:
            h = block(
                h,
                attention_mask=attention_mask,
                position_ids=position_ids
            )[0]

        # Final LayerNorm
        h_norm = self.final_norm(h)           # [batch, seq_len, hidden_dim]

        # Take the hidden state at token‐0 (CLS‐like token) for classification
        cls_token = h_norm[:, 0, :]           # [batch, hidden_dim]
        logits = self.classifier(cls_token)   # [batch, num_labels]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits, labels)

        return logits



# ============================================
# 3) Pipeline Forward and Backward Functions
# ============================================
def pipeline_forward_backward(stages, optimizers, input_chunks, mask_chunks, label_chunks, 
                             device_list, lossy_fraction, use_fp16=False, scaler=None):
    num_stages = len(stages)
    num_microbatches = len(input_chunks)
    activations = {}
    input_grads = {}

    for opt in optimizers:
        opt.zero_grad()

    # Phase 1: Warm-up
    for mb in range(min(num_microbatches, num_stages)):
        for stage_idx in range(min(mb + 1, num_stages)):
            if stage_idx == 0:
                input_ids = input_chunks[mb].to(device_list[0])
                attention_mask = mask_chunks[mb].to(device_list[0])
                
                if use_fp16:
                    with autocast():
                        h, mask, pos_ids = stages[0](input_ids, attention_mask)
                else:
                    h, mask, pos_ids = stages[0](input_ids, attention_mask)
                
                activations[(0, mb)] = (h.detach().requires_grad_(True), mask, pos_ids)
                
            elif stage_idx == num_stages - 1:
                h_in, mask_in, pos_ids_in = activations[(stage_idx - 1, mb)]
                h_in = h_in.to(device_list[stage_idx])
                mask_in = mask_in.to(device_list[stage_idx])
                pos_ids_in = pos_ids_in.to(device_list[stage_idx])
                labels = label_chunks[mb].to(device_list[stage_idx])
                
                if use_fp16:
                    with autocast():
                        loss = stages[stage_idx](h_in, mask_in, pos_ids_in, labels=labels)
                    scaler.scale(loss).backward()
                else:
                    loss = stages[stage_idx](h_in, mask_in, pos_ids_in, labels=labels)
                    loss.backward()
                
                if h_in.grad is not None:
                    grad = zero_some_fraction(h_in.grad.detach(), lossy_fraction)
                    input_grads[(stage_idx - 1, mb)] = grad
                
            else:
                h_in, mask_in, pos_ids_in = activations[(stage_idx - 1, mb)]
                h_in = h_in.to(device_list[stage_idx])
                mask_in = mask_in.to(device_list[stage_idx])
                pos_ids_in = pos_ids_in.to(device_list[stage_idx])
                
                if use_fp16:
                    with autocast():
                        h_out, mask_out, pos_ids_out = stages[stage_idx](h_in, mask_in, pos_ids_in)
                else:
                    h_out, mask_out, pos_ids_out = stages[stage_idx](h_in, mask_in, pos_ids_in)
                
                activations[(stage_idx, mb)] = (h_out.detach().requires_grad_(True), mask_out, pos_ids_out)
    
    # Phase 2: Steady state - 1F1B pattern
    for mb in range(num_stages, num_microbatches):
        # Forward pass for new microbatch
        for stage_idx in range(num_stages):
            if stage_idx == 0:
                input_ids = input_chunks[mb].to(device_list[0])
                attention_mask = mask_chunks[mb].to(device_list[0])
                
                if use_fp16:
                    with autocast():
                        h, mask, pos_ids = stages[0](input_ids, attention_mask)
                else:
                    h, mask, pos_ids = stages[0](input_ids, attention_mask)
                
                # Store the third element (pos_ids) as well
                activations[(0, mb)] = (h.detach().requires_grad_(True), mask, pos_ids)
                
            elif stage_idx == num_stages - 1:
                h_in, mask_in = activations[(stage_idx - 1, mb)]
                h_in = h_in.to(device_list[stage_idx])
                mask_in = mask_in.to(device_list[stage_idx])
                labels = label_chunks[mb].to(device_list[stage_idx])
                
                if use_fp16:
                    with autocast():
                        loss = stages[stage_idx](h_in, mask_in, labels=labels)
                    scaler.scale(loss).backward()
                else:
                    loss = stages[stage_idx](h_in, mask_in, labels=labels)
                    loss.backward()
                
                if h_in.grad is not None:
                    grad = zero_some_fraction(h_in.grad.detach(), lossy_fraction)
                    input_grads[(stage_idx - 1, mb)] = grad
                    
            else:
                h_in, mask_in, pos_ids_in = activations[(stage_idx - 1, mb)]
                h_in = h_in.to(device_list[stage_idx])
                mask_in = mask_in.to(device_list[stage_idx])
                pos_ids_in = pos_ids_in.to(device_list[stage_idx])
                
                if use_fp16:
                    with autocast():
                         h_out, mask_out, pos_ids_out = stages[stage_idx](h_in, mask_in, pos_ids_in)
                else:
                    h_out, mask_out, pos_ids_out = stages[stage_idx](h_in, mask_in, pos_ids_in)
                
                activations[(stage_idx, mb)] = (h_out.detach().requires_grad_(True), mask_out, pos_ids_out)
        
        # Backward pass for oldest microbatch in pipeline
        old_mb = mb - num_stages + 1
        for stage_idx in range(num_stages - 2, -1, -1):
            if stage_idx == 0:
                # First stage backward
                h_stored, mask_stored = activations[(0, old_mb)]
                input_ids = input_chunks[old_mb].to(device_list[0])
                attention_mask = mask_chunks[old_mb].to(device_list[0])
                
                # Recompute forward pass
                if use_fp16:
                    with autocast():
                        # h_recompute, _ = stages[0](input_ids, attention_mask)
                        h_recompute, _, _ = stages[0](input_ids, attention_mask)
                else:
                    # h_recompute, _ = stages[0](input_ids, attention_mask)
                    h_recompute, _, _ = stages[0](input_ids, attention_mask)
                
                # Backward pass
                upstream_grad = input_grads[(0, old_mb)]
                h_recompute.backward(upstream_grad)
                
            else:
                # Middle stage backward
                h_stored, mask_stored = activations[(stage_idx, old_mb)]
                h_in, mask_in, pos_ids_in = activations[(stage_idx - 1, old_mb)]
                h_in = h_in.to(device_list[stage_idx])
                mask_in = mask_in.to(device_list[stage_idx])
                pos_ids_in = pos_ids_in.to(device_list[stage_idx])
                
                # Recompute forward pass
                if use_fp16:
                    with autocast():
                        # h_recompute, _ = stages[stage_idx](h_in, mask_in)
                        h_recompute, _, _ = stages[stage_idx](h_in, mask_in, pos_ids_in)
                else:
                    # h_recompute, _ = stages[stage_idx](h_in, mask_in)
                    h_recompute, _, _ = stages[stage_idx](h_in, mask_in, pos_ids_in)
                
                # Backward pass
                upstream_grad = input_grads[(stage_idx, old_mb)]
                h_recompute.backward(upstream_grad)
                
                # Capture gradient for previous stage
                if h_in.grad is not None:
                    grad = zero_some_fraction(h_in.grad.detach(), lossy_fraction)
                    input_grads[(stage_idx - 1, old_mb)] = grad
    
    # Phase 3: Cool-down - drain the pipeline
    for remaining in range(min(num_stages - 1, num_microbatches)):
        old_mb = num_microbatches - num_stages + 1 + remaining
        if old_mb >= 0:
            for stage_idx in range(num_stages - 2, -1, -1):
                if stage_idx == 0:
                    h_stored, mask_stored = activations[(0, old_mb)]
                    input_ids = input_chunks[old_mb].to(device_list[0])
                    attention_mask = mask_chunks[old_mb].to(device_list[0])
                    
                    if use_fp16:
                        with autocast():
                            # h_recompute, _ = stages[0](input_ids, attention_mask)
                            h_recompute, _, _ = stages[0](input_ids, attention_mask)
                    else:
                        # h_recompute, _ = stages[0](input_ids, attention_mask)
                        h_recompute, _, _ = stages[0](input_ids, attention_mask)
                    
                    upstream_grad = input_grads[(0, old_mb)]
                    h_recompute.backward(upstream_grad)
                    
                else:
                    h_stored, mask_stored = activations[(stage_idx, old_mb)]
                    h_in, mask_in, pos_ids_in = activations[(stage_idx - 1, old_mb)]
                    h_in = h_in.to(device_list[stage_idx])
                    mask_in = mask_in.to(device_list[stage_idx])
                    pos_ids_in = pos_ids_in.to(device_list[stage_idx])
                    
                    if use_fp16:
                        with autocast():
                            # h_recompute, _ = stages[stage_idx](h_in, mask_in)
                            h_recompute, _, _ = stages[stage_idx](h_in, mask_in, pos_ids_in)
                    else:
                        # h_recompute, _ = stages[stage_idx](h_in, mask_in)
                        h_recompute, _, _ = stages[stage_idx](h_in, mask_in, pos_ids_in)
                    
                    upstream_grad = input_grads[(stage_idx, old_mb)]
                    h_recompute.backward(upstream_grad)
                    
                    if h_in.grad is not None:
                        grad = zero_some_fraction(h_in.grad.detach(), lossy_fraction)
                        input_grads[(stage_idx - 1, old_mb)] = grad


# ============================================
# 4) Main Training Function
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
    TOTAL_LAYERS = model_cls.model.config.num_hidden_layers
    assert TOTAL_LAYERS % NUM_GPUS == 0, f"Total layers ({TOTAL_LAYERS}) must be divisible by num_nodes"
    LAYERS_PER_STAGE = TOTAL_LAYERS // NUM_GPUS

    GLOBAL_BATCH_SIZE = args.batch_size
    MICRO_BATCHES = NUM_GPUS
    assert GLOBAL_BATCH_SIZE % MICRO_BATCHES == 0, "Global batch size must be divisible by num_nodes"
    MICRO_BATCH_SIZE = GLOBAL_BATCH_SIZE // MICRO_BATCHES

    LOSSY_FRACTION = args.loss_rate
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    SAVE_EVERY_N_STEPS = args.save_steps
    LOG_EVERY_N_STEPS = args.logging_steps

    DEVICE_LIST = [f"cuda:{i}" for i in range(NUM_GPUS)]
    for dev in DEVICE_LIST:
        if not torch.cuda.is_available():
            raise RuntimeError(f"{dev} is not available")

    # ──────────────────────────────────────────────────────────
    # 3.9 Handle Single‐GPU (NUM_GPUS == 1)
    # ──────────────────────────────────────────────────────────
    if NUM_GPUS == 1:
        # Standard single‐GPU fine‐tuning
        device = torch.device("cuda:0")
        model_cls.to(device)
        optimizer = optim.AdamW(model_cls.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scaler = GradScaler() if args.fp16 else None

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
    # 3.12 Prepare DataLoader
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
    # 3.13 Mixed Precision Setup
    # ──────────────────────────────────────────────────────────
    scaler = GradScaler() if args.fp16 else None

    # ──────────────────────────────────────────────────────────
    # 3.14 Callback Setup
    # ──────────────────────────────────────────────────────────
    callback_args = {
        "report_ttac": dataset_config["report_ttac"],
        "report_file": f"{OUTPUT_DIR_LOCAL}/ttac_report.txt",
        "target_acc": dataset_config["target_acc"],
    }
    callback = MyClassifierCallback(callback_args)

    # ──────────────────────────────────────────────────────────
    # 3.15 Helper: Split batch into microbatches
    # ──────────────────────────────────────────────────────────
    def split_batch_into_microbatches(batch):
        """Split a global batch into microbatches for pipeline processing."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        input_chunks = torch.chunk(input_ids, MICRO_BATCHES, dim=0)
        mask_chunks = torch.chunk(attention_mask, MICRO_BATCHES, dim=0)
        label_chunks = torch.chunk(labels, MICRO_BATCHES, dim=0)
        
        return input_chunks, mask_chunks, label_chunks

    # ──────────────────────────────────────────────────────────
    # 3.16 Pipeline Evaluation Function
    # ──────────────────────────────────────────────────────────
    def evaluate_pipeline():
        """Evaluate the pipeline model on the evaluation dataset."""
        print("Running evaluation...")
        all_predictions = []
        all_labels = []
        total_eval_loss = 0.0
        num_eval_batches = 0
        
        for stage in stages:
            stage.eval()
        
        with torch.no_grad():
            for batch in eval_loader:
                input_chunks, mask_chunks, label_chunks = split_batch_into_microbatches(batch)
                
                # Forward pass through pipeline (evaluation mode)
                for mb_idx in range(len(input_chunks)):
                    # Stage 0
                    input_ids = input_chunks[mb_idx].to(DEVICE_LIST[0])
                    attention_mask = mask_chunks[mb_idx].to(DEVICE_LIST[0])
                    
                    if args.fp16:
                        with autocast():
                            # h, mask = stages[0](input_ids, attention_mask)
                            h, mask, pos_ids = stages[0](input_ids, attention_mask)
                    else:
                        # h, mask = stages[0](input_ids, attention_mask)
                        h, mask, pos_ids = stages[0](input_ids, attention_mask)
                    
                    # Middle stages
                    for i in range(1, NUM_GPUS - 1):
                        h = h.to(DEVICE_LIST[i])
                        mask = mask.to(DEVICE_LIST[i])
                        # if args.fp16:
                        #     with autocast():
                        #         h, mask = stages[i](h, mask)
                        # else:
                        #     h, mask = stages[i](h, mask)
                        if args.fp16:
                            with autocast():
                                h, mask, pos_ids = stages[i](h, mask, pos_ids)
                        else:
                            h, mask, pos_ids = stages[i](h, mask, pos_ids)
                    
                    # Final stage
                    h = h.to(DEVICE_LIST[-1])
                    mask = mask.to(DEVICE_LIST[-1])
                    labels = label_chunks[mb_idx].to(DEVICE_LIST[-1])
                    
                    # if args.fp16:
                    #     with autocast():
                    #         logits = stages[-1](h, mask)
                    #         loss = nn.CrossEntropyLoss()(logits, labels)
                    # else:
                    #     logits = stages[-1](h, mask)
                    #     loss = nn.CrossEntropyLoss()(logits, labels)
                    
                    if args.fp16:
                        with autocast():
                            logits = stages[-1](h, mask, pos_ids)
                            loss = nn.CrossEntropyLoss()(logits, labels)
                    else:
                        logits = stages[-1](h, mask, pos_ids)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                    
                    # Collect predictions and labels
                    predictions = torch.argmax(logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total_eval_loss += loss.item()
                
                num_eval_batches += 1
        
        # Set back to training mode
        for stage in stages:
            stage.train()
        
        # Compute metrics
        avg_eval_loss = total_eval_loss / (num_eval_batches * MICRO_BATCHES)
        eval_metrics = compute_classfication_metrics(all_predictions, all_labels)
        
        return avg_eval_loss, eval_metrics

    # ──────────────────────────────────────────────────────────
    # 3.17 Main Training Loop
    # ──────────────────────────────────────────────────────────
    print(f"Starting pipeline parallel training with {NUM_GPUS} GPUs...")
    print(f"Layers per stage: {LAYERS_PER_STAGE}")
    print(f"Global batch size: {GLOBAL_BATCH_SIZE}, Micro batch size: {MICRO_BATCH_SIZE}")
    print(f"Lossy fraction: {LOSSY_FRACTION}")

    global_step = 0
    best_eval_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n[Pipeline] Epoch {epoch+1}/{NUM_EPOCHS} starting...")
        epoch_loss = 0.0
        num_batches = 0
        
        # Set all stages to training mode
        for stage in stages:
            stage.train()
        
        for batch_idx, batch in enumerate(train_loader):
            # Split global batch into microbatches
            input_chunks, mask_chunks, label_chunks = split_batch_into_microbatches(batch)
            
            # Pipeline forward and backward pass
            pipeline_forward_backward(
                stages=stages,
                optimizers=optimizers,
                input_chunks=input_chunks,
                mask_chunks=mask_chunks,
                label_chunks=label_chunks,
                device_list=DEVICE_LIST,
                lossy_fraction=LOSSY_FRACTION,
                use_fp16=args.fp16,
                scaler=scaler
            )
            
            # Update parameters
            if args.fp16:
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in optimizers:
                    opt.step()
            
            global_step += 1
            
            # Logging
            if global_step % LOG_EVERY_N_STEPS == 0:
                wandb.log({
                    "epoch": epoch,
                    "global_step": global_step,
                    "learning_rate": LEARNING_RATE,
                    "lossy_fraction": LOSSY_FRACTION,
                })
                print(f"Step {global_step}, Epoch {epoch+1}")
            
            # Evaluation
            if hasattr(args, 'eval_steps') and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                eval_loss, eval_metrics = evaluate_pipeline()
                wandb.log({
                    "eval_loss": eval_loss,
                    "eval_accuracy": eval_metrics.get("accuracy", 0.0),
                    "eval_f1": eval_metrics.get("f1", 0.0),
                    "global_step": global_step
                })
                print(f"Evaluation at step {global_step}: Loss={eval_loss:.4f}, Acc={eval_metrics.get('accuracy', 0.0):.4f}")
                
                # Update best accuracy
                current_accuracy = eval_metrics.get("accuracy", 0.0)
                if current_accuracy > best_eval_accuracy:
                    best_eval_accuracy = current_accuracy
                    # Save best model
                    best_ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, "best_model.pt")
                    save_pipeline_checkpoint(stages, optimizers, global_step, epoch, best_ckpt_path)
                    print(f"New best accuracy: {best_eval_accuracy:.4f}, saved to {best_ckpt_path}")
            
            # Save checkpoint
            if global_step % SAVE_EVERY_N_STEPS == 0:
                ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, f"checkpoint-step{global_step}.pt")
                save_pipeline_checkpoint(stages, optimizers, global_step, epoch, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")
            
            num_batches += 1
        
        print(f"[Pipeline] Epoch {epoch+1}/{NUM_EPOCHS} complete.")
        
        # End-of-epoch evaluation
        eval_loss, eval_metrics = evaluate_pipeline()
        wandb.log({
            "epoch_eval_loss": eval_loss,
            "epoch_eval_accuracy": eval_metrics.get("accuracy", 0.0),
            "epoch": epoch + 1
        })
        print(f"End of epoch evaluation: Loss={eval_loss:.4f}, Acc={eval_metrics.get('accuracy', 0.0):.4f}")
        
        # Callback for target accuracy
        if callback:
            callback.on_epoch_end(epoch, eval_metrics)

    # ──────────────────────────────────────────────────────────
    # 3.18 Final Evaluation and Cleanup
    # ──────────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    final_eval_loss, final_eval_metrics = evaluate_pipeline()
    wandb.log({
        "final_eval_loss": final_eval_loss,
        "final_eval_accuracy": final_eval_metrics.get("accuracy", 0.0),
        "best_eval_accuracy": best_eval_accuracy,
    })
    
    print(f"Training completed!")
    print(f"Final evaluation: Loss={final_eval_loss:.4f}, Acc={final_eval_metrics.get('accuracy', 0.0):.4f}")
    print(f"Best accuracy achieved: {best_eval_accuracy:.4f}")
    
    # Save final checkpoint
    final_ckpt_path = os.path.join(OUTPUT_DIR_LOCAL, "final_model.pt")
    save_pipeline_checkpoint(stages, optimizers, global_step, NUM_EPOCHS, final_ckpt_path)
    print(f"Final model saved to {final_ckpt_path}")
    
    wandb.finish()


# ============================================
# 5) Helper: Save Pipeline Checkpoint
# ============================================
def save_pipeline_checkpoint(stages, optimizers, global_step, epoch, save_path):
    """Save checkpoint for pipeline parallel model."""
    checkpoint = {
        "global_step": global_step,
        "epoch": epoch,
        "stage_states": [stage.state_dict() for stage in stages],
        "optimizer_states": [opt.state_dict() for opt in optimizers],
    }
    torch.save(checkpoint, save_path)


# ============================================
# 6) Main Entry Point
# ============================================
if __name__ == "__main__":
    main()