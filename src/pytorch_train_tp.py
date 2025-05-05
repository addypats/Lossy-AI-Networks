#!/usr/bin/env python
import os
import time
import torch
import yaml
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score

from comms import LossyNetwork
from parallel_layers import RowParallelLinear

def train_step(model, inputs, optimizer, network):
    """Single step: forward, backward, apply lossy-network to grads, optimizer."""
    model.train()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    # simulate packet loss on each param.grad
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            mask = network.send(param.grad)
            param.grad = network.receive(param.grad, mask)

    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def evaluate(model, eval_dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return accuracy_score(all_labels, all_preds)

def train_to_accuracy(args):
    # ---- Distributed init ----
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = args.tensor_parallel_size
    group = dist.group.WORLD

    # ---- Load dataset ----
    if args.dataset == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)
        train_ds, eval_ds = ds["train"], ds["validation"]
        num_labels = 2
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # apply sample limits
    if args.max_samples > 0:
        train_ds = train_ds.select(range(min(args.max_samples, len(train_ds))))
        eval_ds = eval_ds.select(range(min(args.max_samples // 5, len(eval_ds))))

    # ---- Model & Tokenizer ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # wrap just the classification head in RowParallelLinear
    orig_head = model.classifier
    model.classifier = RowParallelLinear(orig_head, world_size, group)
    model.to(torch.cuda.current_device())

    # ---- Preprocess (example for winogrande) ----
    def preprocess_winogrande(batch):
        sentence = batch["sentence"]
        opt1, opt2 = batch["option1"], batch["option2"]
        placeholder = "_" if "_" in sentence else "___"
        s1 = sentence.replace(placeholder, opt1)
        s2 = sentence.replace(placeholder, opt2)
        label = 0 if batch["answer"] == '1' else 1
        enc = tokenizer([s1, s2], truncation=True, padding="max_length",
                        max_length=256, return_tensors="pt")
        return {
            'input_ids': enc['input_ids'][label].tolist(),
            'attention_mask': enc['attention_mask'][label].tolist(),
            'labels': label
        }

    train_ds = train_ds.map(preprocess_winogrande,
                            remove_columns=["sentence","option1","option2","answer"])
    eval_ds  = eval_ds.map(preprocess_winogrande,
                           remove_columns=["sentence","option1","option2","answer"])

    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    # ---- Dataloaders with DistributedSampler ----
    train_sampler = DistributedSampler(train_ds,
                                       num_replicas=world_size,
                                       rank=args.local_rank,
                                       shuffle=True)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              sampler=train_sampler)

    eval_sampler = DistributedSampler(eval_ds,
                                      num_replicas=world_size,
                                      rank=args.local_rank,
                                      shuffle=False)
    eval_loader = DataLoader(eval_ds,
                             batch_size=args.batch_size,
                             sampler=eval_sampler)

    # ---- Optimizer & Network ----
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)
    network   = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    # ---- Training loop ----
    best_acc = 0.0
    no_imp   = 0
    step     = 0
    start    = time.time()

    while step < args.max_steps:
        train_sampler.set_epoch(step)  # shuffle differently each epoch
        for batch in tqdm(train_loader, desc=f"Step {step}", disable=(args.local_rank!=0)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, network)
            step += 1

            if step % args.eval_steps == 0 and args.local_rank == 0:
                acc = evaluate(model, eval_loader, model.device)
                elapsed = time.time() - start
                print(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s")

                # save results & early stop
                if acc >= args.target_accuracy:
                    print("Target reached!")
                    torch.save(model.state_dict(), os.path.join(args.output_dir,"model_final.pt"))
                    return
                if acc > best_acc:
                    best_acc, no_imp = acc, 0
                    torch.save(model.state_dict(), os.path.join(args.output_dir,"model_best.pt"))
                else:
                    no_imp +=1
                    if no_imp >= args.patience:
                        print("No improvement; stopping.")
                        return

            if step >= args.max_steps:
                break

    # final save by rank 0
    if args.local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir,"model_final.pt"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tensor-parallel train")
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Provided by torchrun')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                        help='Number of GPUs per node')
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

    # make output_dir per run if desired
    os.makedirs(args.output_dir, exist_ok=True)
    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_to_accuracy(args)
