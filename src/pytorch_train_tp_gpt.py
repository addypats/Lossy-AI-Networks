# # # #!/usr/bin/env python
# # # import os
# # # import time
# # # import json
# # # import torch
# # # import yaml
# # # import numpy as np
# # # import torch.distributed as dist
# # # import torch.nn as nn
# # # from tqdm import tqdm
# # # from datasets import load_dataset
# # # from torch.utils.data import DataLoader
# # # from torch.utils.data.distributed import DistributedSampler
# # # import torch.optim as optim
# # # from transformers import AutoModelForSequenceClassification, AutoTokenizer
# # # from torch.cuda.amp import autocast, GradScaler
# # # from sklearn.metrics import accuracy_score

# # # from comms import LossyNetwork
# # # from parallel_layers import RowParallelLinear

# # # def replace_linears(module: nn.Module, world_size: int, group: dist.ProcessGroup):
# # #     """
# # #     Recursively replace nn.Linear in `module` with RowParallelLinear
# # #     whenever out_features % world_size == 0.
# # #     """
# # #     for name, child in list(module.named_children()):
# # #         if isinstance(child, nn.Linear):
# # #             out_f = child.out_features
# # #             if out_f % world_size == 0:
# # #                 wrapped = RowParallelLinear(child, world_size, group)
# # #                 setattr(module, name, wrapped)
# # #         else:
# # #             replace_linears(child, world_size, group)


# # # def train_step(model, inputs, optimizer, network):
# # #     model.train()
# # #     outputs = model(**inputs)
# # #     loss = outputs.loss
# # #     loss.backward()

# # #     # simulate packet loss on gradients
# # #     for _, param in model.named_parameters():
# # #         if param.requires_grad and param.grad is not None:
# # #             mask = network.send(param.grad)
# # #             param.grad = network.receive(param.grad, mask)

# # #     optimizer.step()
# # #     optimizer.zero_grad()
# # #     return loss.item()


# # # def evaluate(model, eval_loader, device):
# # #     model.eval()
# # #     all_preds, all_labels = [], []
# # #     with torch.no_grad():
# # #         for batch in eval_loader:
# # #             batch = {k: v.to(device) for k, v in batch.items()}
# # #             logits = model(**batch).logits
# # #             preds = torch.argmax(logits, dim=-1).cpu().numpy()
# # #             labels = batch['labels'].cpu().numpy()
# # #             all_preds.extend(preds)
# # #             all_labels.extend(labels)
# # #     return accuracy_score(all_labels, all_preds)


# # # def train_to_accuracy(args):
# # #     # --- Distributed init ---
# # #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
# # #     torch.cuda.set_device(local_rank)
# # #     dist.init_process_group(backend='nccl', init_method='env://')
# # #     world_size = args.tensor_parallel_size
# # #     group = dist.group.WORLD

# # #     # --- Prepare output directory and logs ---
# # #     os.makedirs(args.output_dir, exist_ok=True)
# # #     # Save training arguments
# # #     with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
# # #         yaml.dump(vars(args), f)
# # #     # Open a log file for training metrics
# # #     log_file = os.path.join(args.output_dir, 'training.log')
# # #     log_f = open(log_file, 'w')
# # #     # Prepare metrics list
# # #     metrics = []

# # #     # --- Dataset loading ---
# # #     if args.dataset == "winogrande":
# # #         ds = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)
# # #         train_ds, eval_ds = ds["train"], ds["validation"]
# # #         num_labels = 2
# # #     else:
# # #         raise ValueError(f"Unsupported dataset: {args.dataset}")

# # #     # apply sample limits
# # #     if args.max_samples > 0:
# # #         train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
# # #         eval_ds  = eval_ds.select(range(min(args.max_samples // 5, len(eval_ds))))

# # #     # --- Model & Tokenizer ---
# # #     model = AutoModelForSequenceClassification.from_pretrained(
# # #         args.model_name, num_labels=num_labels, use_auth_token=True
# # #     )
# # #     model.gradient_checkpointing_disable()

# # #     tokenizer = AutoTokenizer.from_pretrained(
# # #         args.model_name, use_auth_token=True
# # #     )
# # #     if tokenizer.pad_token is None:
# # #         tokenizer.pad_token = tokenizer.eos_token
# # #         model.config.pad_token_id = tokenizer.pad_token_id
# # #         model.resize_token_embeddings(len(tokenizer))

# # #     # --- Replace parallel linears ---
# # #     if hasattr(model, 'model'):
# # #         backbone = model.model
# # #     elif hasattr(model, 'transformer'):
# # #         backbone = model.transformer
# # #     else:
# # #         backbone = model
# # #     replace_linears(backbone, world_size, group)
# # #     model.to(torch.cuda.current_device())

# # #     # --- Preprocessing for Winogrande ---
# # #     def preprocess(batch):
# # #         sentence = batch["sentence"]
# # #         o1, o2 = batch["option1"], batch["option2"]
# # #         placeholder = "_" if "_" in sentence else "___"
# # #         s1 = sentence.replace(placeholder, o1)
# # #         s2 = sentence.replace(placeholder, o2)
# # #         label = 0 if batch["answer"] == '1' else 1
# # #         enc = tokenizer([s1, s2], truncation=True, padding="max_length",
# # #                         max_length=args.max_length, return_tensors="pt")
# # #         return {'input_ids': enc['input_ids'][label].tolist(),
# # #                 'attention_mask': enc['attention_mask'][label].tolist(),
# # #                 'labels': label}

# # #     train_ds = train_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
# # #     eval_ds  = eval_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
# # #     train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
# # #     eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# # #     # --- Distributed DataLoaders ---
# # #     train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
# # #                                        rank=local_rank, shuffle=True)
# # #     train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
# # #     eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
# # #                                       rank=local_rank, shuffle=False)
# # #     eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

# # #     # --- AMP setup ---
# # #     use_fp16 = getattr(args, "fp16", False)
# # #     scaler = GradScaler() if use_fp16 else None

# # #     # --- Optimizer & LossyNetwork ---
# # #     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
# # #                             weight_decay=getattr(args, 'weight_decay', 0.0))
# # #     network = LossyNetwork(loss_rate=args.loss_rate)
# # #     network.set_seed(args.seed)

# # #     # --- Training loop ---
# # #     best_acc, no_imp, step = 0.0, 0, 0
# # #     start = time.time()

# # #     while step < args.max_steps:
# # #         train_sampler.set_epoch(step)
# # #         for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}",
# # #                           disable=(local_rank != 0)):
# # #             batch = {k: v.to(model.device) for k, v in batch.items()}
# # #             loss = train_step(model, batch, optimizer, network)

# # #             if use_fp16:
# # #                 with autocast():
# # #                     outputs = model(**batch)
# # #                     loss = outputs.loss
# # #                 scaler.scale(loss).backward()
# # #                 for _, param in model.named_parameters():
# # #                     if param.grad is not None:
# # #                         mask = network.send(param.grad)
# # #                         param.grad = network.receive(param.grad, mask)
# # #                 scaler.step(optimizer)
# # #                 scaler.update()
# # #                 optimizer.zero_grad()

# # #             step += 1
# # #             #if step % args.eval_steps == 0 and local_rank == 0:
# # #             #    acc = evaluate(model, eval_loader, model.device)
# # #             #    elapsed = time.time() - start
# # #             #    line = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
# # #             #    print(line)
# # #             #    log_f.write(line + "\n")
# # #             #    log_f.flush()
# # #             #    metrics.append({'step': step, 'accuracy': acc, 'time': elapsed})

# # #             #    # checkpoint logic
# # #             #    if acc >= args.target_accuracy:
# # #             #        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
# # #             #        # save metrics before exit
# # #             #        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# # #             #            json.dump(metrics, mf, indent=2)
# # #             #        log_f.close()
# # #             #        return
# # #             #    if acc > best_acc:
# # #             #        best_acc, no_imp = acc, 0
# # #             #        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
# # #             #    else:
# # #             #        no_imp += 1
# # #             #        if no_imp >= args.patience:
# # #             #            # save metrics before exit
# # #             #            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# # #             #                json.dump(metrics, mf, indent=2)
# # #             #            log_f.close()
# # #             #            return
            

# # #             # ——— every eval interval, ask rank 0 “are we done?” ———
# # #             if step % args.eval_steps == 0:
# # #                 stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)

# # #                 if args.local_rank == 0:
# # #                     acc = evaluate(model, eval_loader, model.device)
# # #                     print(f"Step {step} | Acc {acc:.4f} | Time {time.time()-start:.1f}s")
# # #                     if acc >= args.target_accuracy:
# # #                         torch.save(model.state_dict(),
# # #                                    os.path.join(args.output_dir, "model_final.pt"))
# # #                         stop_signal.fill_(1)
# # #                     elif acc > best_acc:
# # #                         best_acc, no_imp = acc, 0
# # #                         torch.save(model.state_dict(),
# # #                                    os.path.join(args.output_dir, "model_best.pt"))
# # #                     else:
# # #                         no_imp += 1
# # #                         if no_imp >= args.patience:
# # #                             stop_signal.fill_(1)

# # #                 # Broadcast the single‐byte stop flag from rank 0 to all ranks
# # #                 dist.broadcast(stop_signal, src=0, group=group)

# # #                 # If any rank sees stop_signal==1, we all exit here:
# # #                 if stop_signal.item() == 1:
# # #                     return


# # #             if step >= args.max_steps:
# # #                 break

# # #     # final checkpoint & metrics dump
# # #     if local_rank == 0:
# # #         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
# # #         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# # #             json.dump(metrics, mf, indent=2)
# # #     log_f.close()


# # # if __name__ == "__main__":
# # #     import argparse
# # #     parser = argparse.ArgumentParser(description="Tensor-parallel train with outputs saved")
# # #     parser.add_argument('--local_rank', type=int, default=0,
# # #                         help='Provided by torchrun')
# # #     parser.add_argument('--max_length', type=int, default=256)
# # #     parser.add_argument('--tensor_parallel_size', type=int, default=4,
# # #                         help='GPUs per node')
# # #     parser.add_argument('--model_name', type=str,
# # #                         default='meta-llama/Llama-3.2-1B')
# # #     parser.add_argument('--dataset', type=str,
# # #                         default='winogrande')
# # #     parser.add_argument('--batch_size', type=int, default=8)
# # #     parser.add_argument('--learning_rate', type=float, default=3e-5)
# # #     parser.add_argument('--weight_decay', type=float, default=0.01)
# # #     parser.add_argument('--loss_rate', type=float, default=0.001)
# # #     parser.add_argument('--seed', type=int, default=1234)
# # #     parser.add_argument('--max_samples', type=int, default=0)
# # #     parser.add_argument('--target_accuracy', type=float, default=0.75)
# # #     parser.add_argument('--eval_steps', type=int, default=100)
# # #     parser.add_argument('--patience', type=int, default=3)
# # #     parser.add_argument('--max_steps', type=int, default=100000)
# # #     parser.add_argument('--output_dir', type=str, default='./output')
# # #     args = parser.parse_args()

# # #     # os.makedirs(args.output_dir, exist_ok=True)

# # #     # ——— replicate main.py’s “wrapper” behavior ———
# # #     # ensure output directory exists (your bash is already passing
# # #     # output/${run_id} into --output_dir)
# # #     os.makedirs(args.output_dir, exist_ok=True)

# # #     # dump all of the CLI args to args.yaml for future reference
# # #     import yaml
# # #     with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
# # #         yaml.dump(vars(args), f)
# # #     # ————————————————————————————————————————
    
# # #     torch.manual_seed(args.seed)
# # #     np.random.seed(args.seed)
# # #     train_to_accuracy(args)




# # #!/usr/bin/env python
# # import os
# # import time
# # import json
# # import torch
# # import yaml
# # import numpy as np
# # import torch.distributed as dist
# # import torch.nn as nn
# # from tqdm import tqdm
# # from datasets import load_dataset
# # from torch.utils.data import DataLoader
# # from torch.utils.data.distributed import DistributedSampler
# # import torch.optim as optim
# # from transformers import AutoModelForSequenceClassification, AutoTokenizer
# # from torch.cuda.amp import autocast, GradScaler
# # from sklearn.metrics import accuracy_score

# # from comms import LossyNetwork
# # from parallel_layers import RowParallelLinear

# # def replace_linears(module: nn.Module, world_size: int, group: dist.ProcessGroup):
# #     """
# #     Recursively replace nn.Linear in `module` with RowParallelLinear
# #     whenever out_features % world_size == 0.
# #     """
# #     for name, child in list(module.named_children()):
# #         if isinstance(child, nn.Linear):
# #             out_f = child.out_features
# #             if out_f % world_size == 0:
# #                 wrapped = RowParallelLinear(child, world_size, group)
# #                 setattr(module, name, wrapped)
# #         else:
# #             replace_linears(child, world_size, group)


# # def train_step(model, inputs, optimizer, network):
# #     model.train()
# #     outputs = model(**inputs)
# #     loss = outputs.loss
# #     loss.backward()

# #     # simulate packet loss on gradients
# #     for _, param in model.named_parameters():
# #         if param.requires_grad and param.grad is not None:
# #             mask = network.send(param.grad)
# #             param.grad = network.receive(param.grad, mask)

# #     optimizer.step()
# #     optimizer.zero_grad()
# #     return loss.item()


# # def evaluate(model, eval_loader, device):
# #     model.eval()
# #     all_preds, all_labels = [], []
# #     with torch.no_grad():
# #         for batch in eval_loader:
# #             batch = {k: v.to(device) for k, v in batch.items()}
# #             logits = model(**batch).logits
# #             preds = torch.argmax(logits, dim=-1).cpu().numpy()
# #             labels = batch['labels'].cpu().numpy()
# #             all_preds.extend(preds)
# #             all_labels.extend(labels)
# #     return accuracy_score(all_labels, all_preds)


# # def train_to_accuracy(args):
# #     # --- Distributed init ---
# #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
# #     torch.cuda.set_device(local_rank)
# #     dist.init_process_group(backend='nccl', init_method='env://')
# #     world_size = args.tensor_parallel_size
# #     group = dist.group.WORLD

# #     # --- Prepare output directory and logs ---
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     # Save training arguments
# #     with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
# #         yaml.dump(vars(args), f)
# #     # Ensure compatibility with main.py callback output
# #     open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
# #     # Open a log file for training metrics
# #     log_file = os.path.join(args.output_dir, 'training.log')
# #     log_f = open(log_file, 'w')
# #     # Prepare metrics list
# #     metrics = []

# #     # --- Dataset loading ---
# #     if args.dataset == "winogrande":
# #         ds = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)
# #         train_ds, eval_ds = ds["train"], ds["validation"]
# #         num_labels = 2
# #     else:
# #         raise ValueError(f"Unsupported dataset: {args.dataset}")

# #     # apply sample limits
# #     if args.max_samples > 0:
# #         train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
# #         eval_ds  = eval_ds.select(range(min(args.max_samples // 5, len(eval_ds))))

# #     # --- Model & Tokenizer ---
# #     model = AutoModelForSequenceClassification.from_pretrained(
# #         args.model_name, num_labels=num_labels, use_auth_token=True
# #     )
# #     model.gradient_checkpointing_disable()

# #     tokenizer = AutoTokenizer.from_pretrained(
# #         args.model_name, use_auth_token=True
# #     )
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #         model.config.pad_token_id = tokenizer.pad_token_id
# #         model.resize_token_embeddings(len(tokenizer))

# #     # --- Replace parallel linears ---
# #     if hasattr(model, 'model'):
# #         backbone = model.model
# #     elif hasattr(model, 'transformer'):
# #         backbone = model.transformer
# #     else:
# #         backbone = model
# #     replace_linears(backbone, world_size, group)
# #     model.to(torch.cuda.current_device())

# #     # --- Preprocessing for Winogrande ---
# #     def preprocess(batch):
# #         sentence = batch["sentence"]
# #         o1, o2 = batch["option1"], batch["option2"]
# #         placeholder = "_" if "_" in sentence else "___"
# #         s1 = sentence.replace(placeholder, o1)
# #         s2 = sentence.replace(placeholder, o2)
# #         label = 0 if batch["answer"] == '1' else 1
# #         enc = tokenizer([s1, s2], truncation=True, padding="max_length",
# #                         max_length=args.max_length, return_tensors="pt")
# #         return {'input_ids': enc['input_ids'][label].tolist(),
# #                 'attention_mask': enc['attention_mask'][label].tolist(),
# #                 'labels': label}

# #     train_ds = train_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
# #     eval_ds  = eval_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
# #     train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
# #     eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# #     # --- Distributed DataLoaders ---
# #     train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
# #                                        rank=local_rank, shuffle=True)
# #     train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
# #     eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
# #                                       rank=local_rank, shuffle=False)
# #     eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

# #     # --- AMP setup ---
# #     use_fp16 = getattr(args, "fp16", False)
# #     scaler = GradScaler() if use_fp16 else None

# #     # --- Optimizer & LossyNetwork ---
# #     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
# #                             weight_decay=getattr(args, 'weight_decay', 0.0))
# #     network = LossyNetwork(loss_rate=args.loss_rate)
# #     network.set_seed(args.seed)

# #     # --- Training loop ---
# #     best_acc, no_imp, step = 0.0, 0, 0
# #     start = time.time()

# #     while step < args.max_steps:
# #         train_sampler.set_epoch(step)
# #         for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}",
# #                           disable=(local_rank != 0)):
# #             batch = {k: v.to(model.device) for k, v in batch.items()}
# #             loss = train_step(model, batch, optimizer, network)

# #             if use_fp16:
# #                 with autocast():
# #                     outputs = model(**batch)
# #                     loss = outputs.loss
# #                 scaler.scale(loss).backward()
# #                 for _, param in model.named_parameters():
# #                     if param.grad is not None:
# #                         mask = network.send(param.grad)
# #                         param.grad = network.receive(param.grad, mask)
# #                 scaler.step(optimizer)
# #                 scaler.update()
# #                 optimizer.zero_grad()

# #             step += 1

# #             # ——— every eval interval, ask rank 0 “are we done?” ———
# #             if step % args.eval_steps == 0:
# #                 stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)

# #                 if local_rank == 0:
# #                     acc = evaluate(model, eval_loader, model.device)
# #                     elapsed = time.time() - start
# #                     line = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
# #                     print(line)
# #                     # write to training.log
# #                     log_f.write(line + "\n")
# #                     log_f.flush()
# #                     metrics.append({'step': step, 'accuracy': acc, 'time': elapsed})

# #                     if acc >= args.target_accuracy:
# #                         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
# #                         stop_signal.fill_(1)
# #                     elif acc > best_acc:
# #                         best_acc, no_imp = acc, 0
# #                         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
# #                     else:
# #                         no_imp += 1
# #                         if no_imp >= args.patience:
# #                             stop_signal.fill_(1)

# #                 # Broadcast the single‐byte stop flag from rank 0 to all ranks
# #                 dist.broadcast(stop_signal, src=0, group=group)

# #                 # If any rank sees stop_signal==1, we all exit here:
# #                 if stop_signal.item() == 1:
# #                     # save metrics before exit
# #                     if local_rank == 0:
# #                         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# #                             json.dump(metrics, mf, indent=2)
# #                     log_f.close()
# #                     return

# #             if step >= args.max_steps:
# #                 break

# #     # final checkpoint & metrics dump
# #     if local_rank == 0:
# #         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
# #         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# #             json.dump(metrics, mf, indent=2)
# #     log_f.close()


# # if __name__ == "__main__":
# #     import argparse
# #     parser = argparse.ArgumentParser(description="Tensor-parallel train with outputs saved")
# #     parser.add_argument('--local_rank', type=int, default=0,
# #                         help='Provided by torchrun')
# #     parser.add_argument('--max_length', type=int, default=256)
# #     parser.add_argument('--tensor_parallel_size', type=int, default=4,
# #                         help='GPUs per node')
# #     parser.add_argument('--model_name', type=str,
# #                         default='meta-llama/Llama-3.2-1B')
# #     parser.add_argument('--dataset', type=str,
# #                         default='winogrande')
# #     parser.add_argument('--batch_size', type=int, default=8)
# #     parser.add_argument('--learning_rate', type=float, default=3e-5)
# #     parser.add_argument('--weight_decay', type=float, default=0.01)
# #     parser.add_argument('--loss_rate', type=float, default=0.001)
# #     parser.add_argument('--seed', type=int, default=1234)
# #     parser.add_argument('--max_samples', type=int, default=0)
# #     parser.add_argument('--target_accuracy', type=float, default=0.75)
# #     parser.add_argument('--eval_steps', type=int, default=100)
# #     parser.add_argument('--patience', type=int, default=3)
# #     parser.add_argument('--max_steps', type=int, default=100000)
# #     parser.add_argument('--output_dir', type=str, default='./output')
# #     args = parser.parse_args()

# #     os.makedirs(args.output_dir, exist_ok=True)
# #     # replicate main.py’s wrapper behavior
# #     with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
# #         yaml.dump(vars(args), f)
# #     # ensure ttac_report.txt exists
# #     open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()

# #     torch.manual_seed(args.seed)
# #     np.random.seed(args.seed)
# #     train_to_accuracy(args)



# # #!/usr/bin/env python
# # import os
# # import time
# # import json
# # import torch
# # import yaml
# # import numpy as np
# # import torch.distributed as dist
# # import torch.nn as nn
# # from tqdm import tqdm
# # from datasets import load_dataset
# # from torch.utils.data import DataLoader
# # from torch.utils.data.distributed import DistributedSampler
# # import torch.optim as optim
# # from transformers import AutoModelForSequenceClassification, AutoTokenizer
# # from torch.cuda.amp import autocast, GradScaler
# # from sklearn.metrics import accuracy_score

# # from comms import LossyNetwork
# # from parallel_layers import RowParallelLinear

# # def replace_linears(module: nn.Module, world_size: int, group: dist.ProcessGroup):
# #     """
# #     Recursively replace nn.Linear in `module` with RowParallelLinear
# #     whenever out_features % world_size == 0.
# #     """
# #     for name, child in list(module.named_children()):
# #         if isinstance(child, nn.Linear):
# #             out_f = child.out_features
# #             if out_f % world_size == 0:
# #                 wrapped = RowParallelLinear(child, world_size, group)
# #                 setattr(module, name, wrapped)
# #         else:
# #             replace_linears(child, world_size, group)


# # def train_step(model, inputs, optimizer, network):
# #     model.train()
# #     outputs = model(**inputs)
# #     loss = outputs.loss
# #     loss.backward()

# #     # simulate packet loss on gradients
# #     for _, param in model.named_parameters():
# #         if param.requires_grad and param.grad is not None:
# #             mask = network.send(param.grad)
# #             param.grad = network.receive(param.grad, mask)

# #     optimizer.step()
# #     optimizer.zero_grad()
# #     return loss.item()


# # def evaluate(model, eval_loader, device):
# #     model.eval()
# #     all_preds, all_labels = [], []
# #     with torch.no_grad():
# #         for batch in eval_loader:
# #             batch = {k: v.to(device) for k, v in batch.items()}
# #             logits = model(**batch).logits
# #             preds = torch.argmax(logits, dim=-1).cpu().numpy()
# #             labels = batch['labels'].cpu().numpy()
# #             all_preds.extend(preds)
# #             all_labels.extend(labels)
# #     return accuracy_score(all_labels, all_preds)


# # def train_to_accuracy(args):
# #     # --- Distributed init ---
# #     local_rank = int(os.environ.get("LOCAL_RANK", 0))
# #     torch.cuda.set_device(local_rank)
# #     dist.init_process_group(backend='nccl', init_method='env://')
# #     world_size = args.tensor_parallel_size
# #     group = dist.group.WORLD

# #     # --- Prepare output directory and logs ---
# #     os.makedirs(args.output_dir, exist_ok=True)
# #     # Save training arguments
# #     with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
# #         yaml.dump(vars(args), f)
# #     # Open a log file for training metrics
# #     log_file = os.path.join(args.output_dir, 'training.log')
# #     log_f = open(log_file, 'w')
# #     # Prepare metrics list
# #     metrics = []

# #     # --- Dataset loading ---
# #     if args.dataset == "winogrande":
# #         ds = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)
# #         train_ds, eval_ds = ds["train"], ds["validation"]
# #         num_labels = 2
# #     else:
# #         raise ValueError(f"Unsupported dataset: {args.dataset}")

# #     # apply sample limits
# #     if args.max_samples > 0:
# #         train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
# #         eval_ds  = eval_ds.select(range(min(args.max_samples // 5, len(eval_ds))))

# #     # --- Model & Tokenizer ---
# #     model = AutoModelForSequenceClassification.from_pretrained(
# #         args.model_name, num_labels=num_labels, use_auth_token=True
# #     )
# #     model.gradient_checkpointing_disable()

# #     tokenizer = AutoTokenizer.from_pretrained(
# #         args.model_name, use_auth_token=True
# #     )
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #         model.config.pad_token_id = tokenizer.pad_token_id
# #         model.resize_token_embeddings(len(tokenizer))

# #     # --- Replace parallel linears ---
# #     if hasattr(model, 'model'):
# #         backbone = model.model
# #     elif hasattr(model, 'transformer'):
# #         backbone = model.transformer
# #     else:
# #         backbone = model
# #     replace_linears(backbone, world_size, group)
# #     model.to(torch.cuda.current_device())

# #     # --- Preprocessing for Winogrande ---
# #     def preprocess(batch):
# #         sentence = batch["sentence"]
# #         o1, o2 = batch["option1"], batch["option2"]
# #         placeholder = "_" if "_" in sentence else "___"
# #         s1 = sentence.replace(placeholder, o1)
# #         s2 = sentence.replace(placeholder, o2)
# #         label = 0 if batch["answer"] == '1' else 1
# #         enc = tokenizer([s1, s2], truncation=True, padding="max_length",
# #                         max_length=args.max_length, return_tensors="pt")
# #         return {'input_ids': enc['input_ids'][label].tolist(),
# #                 'attention_mask': enc['attention_mask'][label].tolist(),
# #                 'labels': label}

# #     train_ds = train_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
# #     eval_ds  = eval_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
# #     train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
# #     eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

# #     # --- Distributed DataLoaders ---
# #     train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
# #                                        rank=local_rank, shuffle=True)
# #     train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
# #     eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
# #                                       rank=local_rank, shuffle=False)
# #     eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

# #     # --- AMP setup ---
# #     use_fp16 = getattr(args, "fp16", False)
# #     scaler = GradScaler() if use_fp16 else None

# #     # --- Optimizer & LossyNetwork ---
# #     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
# #                             weight_decay=getattr(args, 'weight_decay', 0.0))
# #     network = LossyNetwork(loss_rate=args.loss_rate)
# #     network.set_seed(args.seed)

# #     # --- Training loop ---
# #     best_acc, no_imp, step = 0.0, 0, 0
# #     start = time.time()

# #     while step < args.max_steps:
# #         train_sampler.set_epoch(step)
# #         for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}",
# #                           disable=(local_rank != 0)):
# #             batch = {k: v.to(model.device) for k, v in batch.items()}
# #             loss = train_step(model, batch, optimizer, network)

# #             if use_fp16:
# #                 with autocast():
# #                     outputs = model(**batch)
# #                     loss = outputs.loss
# #                 scaler.scale(loss).backward()
# #                 for _, param in model.named_parameters():
# #                     if param.grad is not None:
# #                         mask = network.send(param.grad)
# #                         param.grad = network.receive(param.grad, mask)
# #                 scaler.step(optimizer)
# #                 scaler.update()
# #                 optimizer.zero_grad()

# #             step += 1
# #             #if step % args.eval_steps == 0 and local_rank == 0:
# #             #    acc = evaluate(model, eval_loader, model.device)
# #             #    elapsed = time.time() - start
# #             #    line = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
# #             #    print(line)
# #             #    log_f.write(line + "\n")
# #             #    log_f.flush()
# #             #    metrics.append({'step': step, 'accuracy': acc, 'time': elapsed})

# #             #    # checkpoint logic
# #             #    if acc >= args.target_accuracy:
# #             #        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
# #             #        # save metrics before exit
# #             #        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# #             #            json.dump(metrics, mf, indent=2)
# #             #        log_f.close()
# #             #        return
# #             #    if acc > best_acc:
# #             #        best_acc, no_imp = acc, 0
# #             #        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
# #             #    else:
# #             #        no_imp += 1
# #             #        if no_imp >= args.patience:
# #             #            # save metrics before exit
# #             #            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# #             #                json.dump(metrics, mf, indent=2)
# #             #            log_f.close()
# #             #            return
            

# #             # ——— every eval interval, ask rank 0 “are we done?” ———
# #             if step % args.eval_steps == 0:
# #                 stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)

# #                 if args.local_rank == 0:
# #                     acc = evaluate(model, eval_loader, model.device)
# #                     print(f"Step {step} | Acc {acc:.4f} | Time {time.time()-start:.1f}s")
# #                     if acc >= args.target_accuracy:
# #                         torch.save(model.state_dict(),
# #                                    os.path.join(args.output_dir, "model_final.pt"))
# #                         stop_signal.fill_(1)
# #                     elif acc > best_acc:
# #                         best_acc, no_imp = acc, 0
# #                         torch.save(model.state_dict(),
# #                                    os.path.join(args.output_dir, "model_best.pt"))
# #                     else:
# #                         no_imp += 1
# #                         if no_imp >= args.patience:
# #                             stop_signal.fill_(1)

# #                 # Broadcast the single‐byte stop flag from rank 0 to all ranks
# #                 dist.broadcast(stop_signal, src=0, group=group)

# #                 # If any rank sees stop_signal==1, we all exit here:
# #                 if stop_signal.item() == 1:
# #                     return


# #             if step >= args.max_steps:
# #                 break

# #     # final checkpoint & metrics dump
# #     if local_rank == 0:
# #         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
# #         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
# #             json.dump(metrics, mf, indent=2)
# #     log_f.close()


# # if __name__ == "__main__":
# #     import argparse
# #     parser = argparse.ArgumentParser(description="Tensor-parallel train with outputs saved")
# #     parser.add_argument('--local_rank', type=int, default=0,
# #                         help='Provided by torchrun')
# #     parser.add_argument('--max_length', type=int, default=256)
# #     parser.add_argument('--tensor_parallel_size', type=int, default=4,
# #                         help='GPUs per node')
# #     parser.add_argument('--model_name', type=str,
# #                         default='meta-llama/Llama-3.2-1B')
# #     parser.add_argument('--dataset', type=str,
# #                         default='winogrande')
# #     parser.add_argument('--batch_size', type=int, default=8)
# #     parser.add_argument('--learning_rate', type=float, default=3e-5)
# #     parser.add_argument('--weight_decay', type=float, default=0.01)
# #     parser.add_argument('--loss_rate', type=float, default=0.001)
# #     parser.add_argument('--seed', type=int, default=1234)
# #     parser.add_argument('--max_samples', type=int, default=0)
# #     parser.add_argument('--target_accuracy', type=float, default=0.75)
# #     parser.add_argument('--eval_steps', type=int, default=100)
# #     parser.add_argument('--patience', type=int, default=3)
# #     parser.add_argument('--max_steps', type=int, default=100000)
# #     parser.add_argument('--output_dir', type=str, default='./output')
# #     args = parser.parse_args()

# #     # os.makedirs(args.output_dir, exist_ok=True)

# #     # ——— replicate main.py’s “wrapper” behavior ———
# #     # ensure output directory exists (your bash is already passing
# #     # output/${run_id} into --output_dir)
# #     os.makedirs(args.output_dir, exist_ok=True)

# #     # dump all of the CLI args to args.yaml for future reference
# #     import yaml
# #     with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
# #         yaml.dump(vars(args), f)
# #     # ————————————————————————————————————————
    
# #     torch.manual_seed(args.seed)
# #     np.random.seed(args.seed)
# #     train_to_accuracy(args)



# #!/usr/bin/env python
# import os
# import time
# import json
# import torch
# import yaml
# import numpy as np
# import torch.distributed as dist
# import torch.nn as nn
# from tqdm import tqdm
# from datasets import load_dataset
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import torch.optim as optim
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from torch.cuda.amp import autocast, GradScaler
# from sklearn.metrics import accuracy_score
# import wandb

# from comms import LossyNetwork
# from data import get_dataset
# from parallel_layers_gpt import RowParallelLinear

# def replace_linears(module: nn.Module, world_size: int, group: dist.ProcessGroup):
#     """
#     Recursively replace nn.Linear in `module` with RowParallelLinear
#     whenever out_features % world_size == 0.
#     """
#     for name, child in list(module.named_children()):
#         if isinstance(child, nn.Linear):
#             out_f = child.out_features
#             if out_f % world_size == 0:
#                 wrapped = RowParallelLinear(child, world_size, group)
#                 setattr(module, name, wrapped)
#         else:
#             replace_linears(child, world_size, group)


# def train_step(model, inputs, optimizer, network):
#     model.train()
#     outputs = model(**inputs)
#     loss = outputs.loss
#     loss.backward()

#     # simulate packet loss on gradients
#     for _, param in model.named_parameters():
#         if param.requires_grad and param.grad is not None:
#             mask = network.send(param.grad)
#             param.grad = network.receive(param.grad, mask)

#     optimizer.step()
#     optimizer.zero_grad()
#     return loss.item()


# def evaluate(model, eval_loader, device):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in eval_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             logits = model(**batch).logits
#             preds = torch.argmax(logits, dim=-1).cpu().numpy()
#             labels = batch['labels'].cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels)
#     return accuracy_score(all_labels, all_preds)


# def train_to_accuracy(args):
#     # --- Distributed init ---
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend='nccl', init_method='env://')
#     world_size = args.tensor_parallel_size
#     group = dist.group.WORLD

#     # --- Load dataset config ---
#     with open("src/config.yaml") as cf:
#         dataset_config = yaml.safe_load(cf)[args.dataset]

#     num_labels     = dataset_config['num_labels']
#     args.target_accuracy = dataset_config['target_acc']
#     report_ttac    = dataset_config.get('report_ttac', [])

#     # --- Prepare output directory and logs ---
#     os.makedirs(args.output_dir, exist_ok=True)
#     # Save training arguments
#     with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
#         yaml.dump(vars(args), f)
#     # Ensure compatibility with main.py callback output
#     open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
#     # Open a log file for training metrics
#     log_file = os.path.join(args.output_dir, 'training.log')
#     log_f = open(log_file, 'w')
#     # Prepare metrics list
#     metrics = []

#     # # --- Dataset loading ---
#     # if args.dataset == "winogrande":
#     #     ds = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)
#     #     train_ds, eval_ds = ds["train"], ds["validation"]
#     #     num_labels = 2
#     # else:
#     #     raise ValueError(f"Unsupported dataset: {args.dataset}")

#     # # apply sample limits
#     # if args.max_samples > 0:
#     #     train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
#     #     eval_ds  = eval_ds.select(range(min(args.max_samples // 5, len(eval_ds))))


#     # --- Model & Tokenizer ---
#     model = AutoModelForSequenceClassification.from_pretrained(
#         args.model_name, num_labels=num_labels, use_auth_token=True
#     )
#     model.gradient_checkpointing_disable()

#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name, use_auth_token=True
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         model.config.pad_token_id = tokenizer.pad_token_id
#         model.resize_token_embeddings(len(tokenizer))

#     # --- Replace parallel linears ---
#     if hasattr(model, 'model'):
#         backbone = model.model
#     elif hasattr(model, 'transformer'):
#         backbone = model.transformer
#     else:
#         backbone = model
#     replace_linears(backbone, world_size, group)
#     model.to(torch.cuda.current_device())

    
#     # # --- Preprocessing for Winogrande ---
#     # def preprocess(batch):
#     #     sentence = batch["sentence"]
#     #     o1, o2 = batch["option1"], batch["option2"]
#     #     placeholder = "_" if "_" in sentence else "___"
#     #     s1 = sentence.replace(placeholder, o1)
#     #     s2 = sentence.replace(placeholder, o2)
#     #     label = 0 if batch["answer"] == '1' else 1
#     #     enc = tokenizer([s1, s2], truncation=True, padding="max_length",
#     #                     max_length=args.max_length, return_tensors="pt")
#     #     return {'input_ids': enc['input_ids'][label].tolist(),
#     #             'attention_mask': enc['attention_mask'][label].tolist(),
#     #             'labels': label}

#     # train_ds = train_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
#     # eval_ds  = eval_ds.map(preprocess, remove_columns=["sentence","option1","option2","answer"])
#     # train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
#     # eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

#     # --- Dataset loading via data.py ---
#     train_ds, eval_ds = get_dataset(args, tokenizer)
#     train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
#     eval_ds.set_format(type='torch',   columns=['input_ids','attention_mask','labels'])

#     # --- Distributed DataLoaders ---
#     train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
#                                        rank=local_rank, shuffle=True)
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
#     eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
#                                       rank=local_rank, shuffle=False)
#     eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

#     # --- AMP setup ---
#     use_fp16 = getattr(args, "fp16", False)
#     scaler = GradScaler() if use_fp16 else None

#     # --- Optimizer & LossyNetwork ---
#     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
#                             weight_decay=getattr(args, 'weight_decay', 0.0))
#     network = LossyNetwork(loss_rate=args.loss_rate)
#     network.set_seed(args.seed)

#     # --- Training loop ---
#     best_acc, no_imp, step = 0.0, 0, 0
#     start = time.time()

#     while step < args.max_steps:
#         train_sampler.set_epoch(step)
#         for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}",
#                           disable=(local_rank != 0)):
#             batch = {k: v.to(model.device) for k, v in batch.items()}
#             loss = train_step(model, batch, optimizer, network)

#             if use_fp16:
#                 with autocast():
#                     outputs = model(**batch)
#                     loss = outputs.loss
#                 scaler.scale(loss).backward()
#                 for _, param in model.named_parameters():
#                     if param.grad is not None:
#                         mask = network.send(param.grad)
#                         param.grad = network.receive(param.grad, mask)
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#             step += 1

#             # ——— every eval interval, ask rank 0 “are we done?” ———
#             if step % args.eval_steps == 0:
#                 stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)

#                 if local_rank == 0:
#                     acc = evaluate(model, eval_loader, model.device)
#                     elapsed = time.time() - start
#                     line = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
#                     print(line)
#                     # write to training.log
#                     log_f.write(line + "\n")
#                     log_f.flush()
#                     metrics.append({'step': step, 'accuracy': acc, 'time': elapsed})

#                     if acc >= args.target_accuracy:
#                         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
#                         stop_signal.fill_(1)
#                     elif acc > best_acc:
#                         best_acc, no_imp = acc, 0
#                         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
#                     else:
#                         no_imp += 1
#                         if no_imp >= args.patience:
#                             stop_signal.fill_(1)

#                 # Broadcast the single‐byte stop flag from rank 0 to all ranks
#                 dist.broadcast(stop_signal, src=0, group=group)

#                 # If any rank sees stop_signal==1, we all exit here:
#                 if stop_signal.item() == 1:
#                     # save metrics before exit
#                     if local_rank == 0:
#                         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
#                             json.dump(metrics, mf, indent=2)
#                     log_f.close()
#                     return

#             if step >= args.max_steps:
#                 break

#     # final checkpoint & metrics dump
#     if local_rank == 0:
#         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
#         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
#             json.dump(metrics, mf, indent=2)
#     log_f.close()


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Tensor-parallel train with outputs saved")
#     parser.add_argument('--local_rank', type=int, default=0,
#                         help='Provided by torchrun')
#     parser.add_argument('--max_length', type=int, default=256)
#     parser.add_argument('--tensor_parallel_size', type=int, default=4,
#                         help='GPUs per node')
#     parser.add_argument('--model_name', type=str,
#                         default='meta-llama/Llama-3.2-1B')
#     parser.add_argument('--dataset', type=str,
#                         default='winogrande')
#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument('--learning_rate', type=float, default=3e-5)
#     parser.add_argument('--weight_decay', type=float, default=0.01)
#     parser.add_argument('--loss_rate', type=float, default=0.001)
#     parser.add_argument("--fp16", action="store_true", help="Enable mixed-precision training (wraps forward/backward in autocast + GradScaler)")
#     parser.add_argument('--seed', type=int, default=1234)
#     parser.add_argument('--max_samples', type=int, default=0)
#     parser.add_argument('--target_accuracy', type=float, default=0.75)
#     parser.add_argument('--eval_steps', type=int, default=100)
#     parser.add_argument('--patience', type=int, default=3)
#     parser.add_argument('--max_steps', type=int, default=100000)
#     parser.add_argument('--output_dir', type=str, default='./output')
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
#     # replicate main.py’s wrapper behavior
#     with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
#         yaml.dump(vars(args), f)
#     # ensure ttac_report.txt exists
#     open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()

#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     train_to_accuracy(args)



#!/usr/bin/env python
import os
import time
import json
import torch
import yaml
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import wandb  # ← wandb import

from comms import LossyNetwork, GillbertElliotLossyNetwork
from data import get_dataset
# from parallel_layers_gpt import RowParallelLinear, ColumnParallelLinear
from parallel_layers_gpt import RowParallelLinear
from transformers.models.gpt2.modeling_gpt2 import Conv1D

# Original replace_linears with only RowParallel
def replace_linears(module: nn.Module, world_size: int, group: dist.ProcessGroup):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            out_f = child.out_features
            if out_f % world_size == 0:
                wrapped = RowParallelLinear(child, world_size, group)
                setattr(module, name, wrapped)
        else:
            replace_linears(child, world_size, group)


# New replace_linears with both
# def replace_linears(module, world_size, group):
#     for name, child in list(module.named_children()):
#         if isinstance(child, (nn.Linear, Conv1D)):
#             in_f, out_f = child.in_features, child.out_features
#             if in_f % world_size == 0:
#                 wrapped = ColumnParallelLinear(child, world_size, group)
#             elif out_f % world_size == 0:
#                 wrapped = RowParallelLinear(child, world_size, group)
#             else:
#                 continue
#             setattr(module, name, wrapped)
#         else:
#             replace_linears(child, world_size, group)

# def replace_linears(module, world_size, group):
#     """
#     Replace linear layers with row-parallel versions only.
#     This ensures consistent tensor shapes throughout the model.
#     """
#     for name, child in list(module.named_children()):
#         if isinstance(child, (nn.Linear, Conv1D)):
#             # grab weight shape
#             W = child.weight.data
#             if isinstance(child, Conv1D):
#                 # Conv1D.weight is [in, out]
#                 in_f, out_f = W.shape
#             else:
#                 # Linear.weight is [out, in]
#                 out_f, in_f = W.shape

#             # Only use row-parallel (shard the output dim) for consistency
#             if out_f % world_size == 0:
#                 wrapped = RowParallelLinear(child, world_size, group)
#                 setattr(module, name, wrapped)
#                 print(f"Parallelized layer {name}: [{out_f}, {in_f}] -> RowParallel")
#             else:
#                 print(f"Skipped layer {name}: output dim {out_f} not divisible by world_size {world_size}")

#         else:
#             replace_linears(child, world_size, group)

def train_step(model, inputs, optimizer, network):
    model.train()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            mask = network.send(param.grad)
            param.grad = network.receive(param.grad, mask)

    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

# def evaluate(model, eval_loader, device):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in eval_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             logits = model(**batch).logits
#             preds = torch.argmax(logits, dim=-1).cpu().numpy()
#             labels = batch['labels'].cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels)
#     return accuracy_score(all_labels, all_preds)

def evaluate(model, eval_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    all_losses = []
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch['labels']
            loss = loss_fn(logits, labels)
            all_losses.append(loss.item())

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(all_losses)
    perplexity = np.exp(avg_loss)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, avg_loss, perplexity

def train_to_accuracy(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://')
    dist.init_process_group(backend='gloo', init_method='env://')
    print(f"[Rank {local_rank}] world_size = {dist.get_world_size()}")
    world_size = args.tensor_parallel_size
    
    # Validate that actual world size matches expected tensor parallel size
    actual_world_size = dist.get_world_size()
    if actual_world_size != world_size:
        print(f"Warning: actual world_size ({actual_world_size}) != tensor_parallel_size ({world_size})")
        world_size = actual_world_size
        
    group = dist.group.WORLD

    with open("src/config.yaml") as cf:
        dataset_config = yaml.safe_load(cf)[args.dataset]

    num_labels     = dataset_config['num_labels']
    args.target_accuracy = dataset_config['target_acc']
    report_ttac    = dataset_config.get('report_ttac', [])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
    log_file = os.path.join(args.output_dir, 'training.log')
    log_f = open(log_file, 'w')
    metrics = []

    # — wandb initialization —
    # if local_rank == 0:
    #     run_id = os.path.basename(os.path.abspath(args.output_dir))
    #     wandb.init(
    #         project="your_project_name",  # ← Replace with your actual project name
    #         name=run_id,
    #         config=vars(args)
    #     )
    if local_rank == 0 and args.wandb:
        run_id = os.path.basename(os.path.abspath(args.output_dir))
        wandb.init(
            project="your_project_name",
            name=run_id,
            config=vars(args),
            tags=[args.model_name, args.dataset, args.loss_type],
            group=args.dataset
        )


    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, use_auth_token=True
    )
    model.gradient_checkpointing_disable()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_auth_token=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, 'model'):
        backbone = model.model
    elif hasattr(model, 'transformer'):
        backbone = model.transformer
    else:
        backbone = model
    replace_linears(backbone, world_size, group)
    # print(backbone)
    model.to(torch.cuda.current_device())

    train_ds, eval_ds = get_dataset(args, tokenizer)
    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    eval_ds.set_format(type='torch',   columns=['input_ids','attention_mask','labels'])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
                                      rank=local_rank, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

    use_fp16 = getattr(args, "fp16", False)
    scaler = GradScaler() if use_fp16 else None

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=getattr(args, 'weight_decay', 0.0))
    # network = LossyNetwork(loss_rate=args.loss_rate)
    
    # Custom loss rate with Bursty Losses (GilbertElliot)
    if args.loss_type == 'ber':
        network = LossyNetwork(loss_rate=args.loss_rate)
    elif args.loss_type == 'g-e':
        import pandas as pd
        configs = pd.read_csv('g_e_params.csv')
        ge_config = configs[configs['id'] == args.ge_config].iloc[0]
        network = GillbertElliotLossyNetwork(
            p_bg = ge_config[' pbg'],
            p_gb = ge_config[' pgb'],
            good_loss_rate = ge_config[' lrg'],
            bad_loss_rate = ge_config[' lrb'],
            args=args
        )
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")

    network.set_seed(args.seed)

    best_acc, no_imp, step = 0.0, 0, 0
    # target_hit_counter = 0  # count how many times accuracy exceeds target
    last_accuracies = []  # For tracking the last 5 evaluation accuracies
    start = time.time()

    while step < args.max_steps:
        train_sampler.set_epoch(step)
        for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}",
                          disable=(local_rank != 0)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            if use_fp16:
                # FP16 training path
                model.train()
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                for _, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        mask = network.send(param.grad)
                        param.grad = network.receive(param.grad, mask)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # Regular training path
                loss = train_step(model, batch, optimizer, network)

            step += 1

            # ----------- TRAINING METRIC LOGGING --------------
            if local_rank == 0 and args.wandb:
                current_lr = optimizer.param_groups[0]['lr']
                total_grad_norm = 0.0
                count = 0
                for _, param in model.named_parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                        count += 1
                grad_norm = (total_grad_norm ** 0.5) if count > 0 else 0.0

                if isinstance(loss, torch.Tensor):
                    train_loss = loss.detach().cpu().item()
                else:
                    train_loss = float(loss)

                wandb.log({
                    'train/loss': train_loss,
                    'train/grad_norm': grad_norm,
                    'train/learning_rate': current_lr,
                    'train/step': step
                })
            # --------------------------------------------------


            if step % args.eval_steps == 0:
                stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)

                if local_rank == 0:
                    # acc = evaluate(model, eval_loader, model.device)
                    acc, val_loss, perplexity = evaluate(model, eval_loader, model.device)
                    elapsed = time.time() - start
                    line = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
                    print(line)
                    log_f.write(line + "\n")
                    log_f.flush()
                    # metrics.append({'step': step, 'accuracy': acc, 'time': elapsed})
                    # wandb.log({'accuracy': acc, 'step': step, 'time': elapsed})
                    
                    metrics.append({
                        'step': step,
                        'accuracy': acc,
                        'loss': val_loss,
                        'perplexity': perplexity,
                        'time': elapsed
                    })
                    if args.wandb and local_rank == 0:
                        wandb.log({
                            'eval/accuracy': acc,
                            'eval/loss': val_loss,
                            'eval/perplexity': perplexity,
                            'eval/step': step,
                            'eval/time': elapsed
                        })

                    
                    # Update list of last 5 accuracies
                    last_accuracies.append(acc)
                    if len(last_accuracies) > 5:
                        last_accuracies.pop(0)

                    # Compute running average
                    avg_acc = sum(last_accuracies) / len(last_accuracies)
                    print(f"Last {len(last_accuracies)} accuracies: {last_accuracies} | Running Avg: {avg_acc:.4f}")


                    # Original code for hit rate to 5 accuracies.
                    
                    # if acc >= args.target_accuracy:
                        
                    #     # Original code
                    #     # torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
                    #     # stop_signal.fill_(1)
                        
                    #     # New code for counting the acc 5 times
                    #     target_hit_counter += 1
                    #     print(f"Target accuracy hit {target_hit_counter} time(s).")
                    #     torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_hit_{target_hit_counter}.pt"))

                    #     if target_hit_counter >= 5:
                    #         print(f"Target accuracy hit 5 times. Stopping training.")
                    #         stop_signal.fill_(1)
                    # elif acc > best_acc:
                    #     best_acc, no_imp = acc, 0
                    #     torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    # else:
                    #     no_imp += 1
                    #     if no_imp >= args.patience:
                    #         stop_signal.fill_(1)
                    
                    # Stop if running average >= target
                    if len(last_accuracies) == 5 and avg_acc >= args.target_accuracy:
                        print(f"Running average of last 5 evaluations reached target accuracy {args.target_accuracy}. Stopping.")
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
                        stop_signal.fill_(1)

                    # Otherwise, track best single accuracy for saving purposes
                    elif acc > best_acc:
                        best_acc, no_imp = acc, 0
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    else:
                        no_imp += 1
                        if no_imp >= args.patience:
                            stop_signal.fill_(1)

                dist.broadcast(stop_signal, src=0, group=group)
                if stop_signal.item() == 1:
                    if local_rank == 0:
                        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
                            json.dump(metrics, mf, indent=2)
                        wandb.finish()
                    log_f.close()
                    return


            # # New train till steps not accuracy function
            # if step % args.eval_steps == 0:
            #     # fixed-budget mode: just evaluate & log (no early stopping)
            #     if local_rank == 0:
            #         acc     = evaluate(model, eval_loader, model.device)
            #         elapsed = time.time() - start
            #         line    = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
            #         print(line)
            #         log_f.write(line + "\n")
            #         log_f.flush()
            #         metrics.append({'step': step, 'accuracy': acc, 'time': elapsed})
            #         wandb.log({'accuracy': acc, 'step': step, 'time': elapsed})

            # if step >= args.max_steps:
            #     break

    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
            json.dump(metrics, mf, indent=2)
        wandb.finish()
    log_f.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tensor-parallel train with outputs saved")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=256)
    
    # GilbertElliot Loss Params
    parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e'],
                        help='Type of packet loss simulation: "ber" for Bernoulli, "g-e" for Gilbert-Elliott')
    parser.add_argument('--ge_config', type=str, default='default',
                        help='Configuration ID for Gilbert-Elliott simulation (used in g_e_params.csv)')

    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--dataset', type=str, default='winogrande')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss_rate', type=float, default=0.001)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--target_accuracy', type=float, default=0.75)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--wandb', action='store_true', help="Enable Weights & Biases logging")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_to_accuracy(args)
