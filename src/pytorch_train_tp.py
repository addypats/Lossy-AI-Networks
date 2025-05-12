#!/usr/bin/env python
import os
import time
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
from sklearn.metrics import accuracy_score
import torch.distributed as dist

from comms import LossyNetwork
from parallel_layers import RowParallelLinear

def replace_linears(module: nn.Module, world_size: int, group: dist.ProcessGroup):
    """
    Recursively replace nn.Linear in `module` with RowParallelLinear
    whenever out_features % world_size == 0.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            out_f = child.out_features
            if out_f % world_size == 0:
                # perform the wrap
                wrapped = RowParallelLinear(child, world_size, group)
                setattr(module, name, wrapped)
        else:
            replace_linears(child, world_size, group)

def train_step(model, inputs, optimizer, network):
    model.train()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    # simulate packet loss on gradients
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            mask = network.send(param.grad)
            param.grad = network.receive(param.grad, mask)

    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def evaluate(model, eval_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return accuracy_score(all_labels, all_preds)

def train_to_accuracy(args):
    # --- Distributed init ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://')
    dist.init_process_group(
    backend='nccl',
    init_method='env://',
    world_size=args.tensor_parallel_size,
    rank=local_rank,
    # pass device_id so NCCL knows exactly which GPU this rank owns
    group_backend_options={"device_id": local_rank}
    )
    world_size = args.tensor_parallel_size
    group = dist.group.WORLD

    # --- Dataset loading (only winogrande shown; extend as needed) ---
    if args.dataset == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)
        train_ds, eval_ds = ds["train"], ds["validation"]
        num_labels = 2
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # apply sample limits
    if args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        eval_ds  = eval_ds.select(range(min(args.max_samples // 5, len(eval_ds))))

    # --- Model & Tokenizer ---
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=num_labels,
        use_auth_token=True
    )

    model.gradient_checkpointing_disable()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_auth_token=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id  = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    # --- Replace all large linears in transformer blocks ---
    # (model.model is the LlamaModel inside LlamaForSequenceClassification)
    replace_linears(model.model, world_size, group)
    model.to(torch.cuda.current_device())

    # --- Preprocessing for Winogrande (as before) ---
    def preprocess(batch):
        sentence = batch["sentence"]
        o1, o2 = batch["option1"], batch["option2"]
        placeholder = "_" if "_" in sentence else "___"
        s1 = sentence.replace(placeholder, o1)
        s2 = sentence.replace(placeholder, o2)
        label = 0 if batch["answer"] == '1' else 1
        enc = tokenizer([s1, s2],
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_length,
                        return_tensors="pt")
        return {
            'input_ids': enc['input_ids'][label].tolist(),
            'attention_mask': enc['attention_mask'][label].tolist(),
            'labels': label
        }

    train_ds = train_ds.map(preprocess,
                            remove_columns=["sentence","option1","option2","answer"])
    eval_ds  = eval_ds.map(preprocess,
                           remove_columns=["sentence","option1","option2","answer"])

    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    # --- DistributedSamplers and Dataloaders ---
    train_sampler = DistributedSampler(train_ds,
                                       num_replicas=world_size,
                                       rank=local_rank,
                                       shuffle=True)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              sampler=train_sampler)

    eval_sampler = DistributedSampler(eval_ds,
                                      num_replicas=world_size,
                                      rank=local_rank,
                                      shuffle=False)
    eval_loader = DataLoader(eval_ds,
                             batch_size=args.batch_size,
                             sampler=eval_sampler)

    # set up AMP
    use_fp16 = getattr(args, "fp16", False)
    scaler = GradScaler() if use_fp16 else None

    # --- Optimizer & LossyNetwork ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    # --- Training loop ---
    best_acc = 0.0
    no_imp   = 0
    step     = 0
    start    = time.time()

    dist.barrier()

    while step < args.max_steps:
        train_sampler.set_epoch(step)
        for batch in tqdm(train_loader, desc=f"Rank {args.local_rank} Step {step}", disable=(args.local_rank != 0)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, network)

            # Mixed-precision forward/backward:
            if use_fp16:
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()

                # mask grads, just like before:
                for _, param in model.named_parameters():
                    if param.grad is not None:
                        mask = network.send(param.grad)
                        param.grad = network.receive(param.grad, mask)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                loss = train_step(model, batch, optimizer, network)

            step += 1

            if step % args.eval_steps == 0 and args.local_rank == 0:
                acc = evaluate(model, eval_loader, model.device)
                elapsed = time.time() - start
                print(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s")

                if acc >= args.target_accuracy:
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
                    return
                if acc > best_acc:
                    best_acc, no_imp = acc, 0
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                else:
                    no_imp += 1
                    if no_imp >= args.patience:
                        return
            if step >= args.max_steps:
                break

    if args.local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tensor-parallel train")
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Provided by torchrun')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='GPUs per node')
    parser.add_argument('--model_name', type=str,
                        default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--dataset', type=str,
                        default='winogrande')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--target_accuracy', type=float, default=0.75)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_to_accuracy(args)
