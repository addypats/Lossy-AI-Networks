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
import wandb

from parallel_layers_gpt_1 import parallelize_gpt2
from comms import LossyNetwork, GillbertElliotLossyNetwork
from data import get_dataset


def train_step(model, inputs, optimizer, network, use_fp16=False, scaler=None):
    model.train()
    if use_fp16:
        with autocast():
            outputs = model(**inputs)
            loss = outputs.loss
        scaler.scale(loss).backward()
    else:
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()

    # apply custom lossy network to gradients
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            mask = network.send(param.grad)
            param.grad = network.receive(param.grad, mask)

    if use_fp16:
        scaler.step(optimizer)
        scaler.update()
    else:
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
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='gloo', init_method='env://')
    world_size = args.tensor_parallel_size
    group = dist.group.WORLD

    # Load dataset configuration
    with open("src/config.yaml") as cf:
        dataset_config = yaml.safe_load(cf)[args.dataset]

    num_labels = dataset_config['num_labels']
    args.target_accuracy = dataset_config['target_acc']
    report_ttac = dataset_config.get('report_ttac', [])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    open(os.path.join(args.output_dir, "ttac_report.txt"), "a").close()
    log_file = os.path.join(args.output_dir, "training.log")
    log_f = open(log_file, "w")
    metrics = []

    # WandB initialization (only on rank 0)
    if local_rank == 0:
        run_id = os.path.basename(os.path.abspath(args.output_dir))
        wandb.init(
            project="your_project_name",
            name=run_id,
            config=vars(args)
        )

    # Model and tokenizer
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

    # Apply Megatron-style tensor parallelism
    parallelize_gpt2(model, world_size=world_size, group=group)
    model.to(torch.cuda.current_device())

    # Prepare datasets and loaders
    train_ds, eval_ds = get_dataset(args, tokenizer)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
                                      rank=local_rank, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

    # FP16 settings
    use_fp16 = args.fp16
    scaler = GradScaler() if use_fp16 else None

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=getattr(args, "weight_decay", 0.0))

    # Custom Lossy Network setup
    if args.loss_type == "ber":
        network = LossyNetwork(args)
    elif args.loss_type == "g-e":
        import pandas as pd
        configs = pd.read_csv("g_e_params.csv")
        ge_config = configs[configs['id'] == args.ge_config].iloc[0]
        network = GillbertElliotLossyNetwork(
            p_bg=ge_config[' pbg'],
            p_gb=ge_config[' pgb'],
            good_loss_rate=ge_config[' lrg'],
            bad_loss_rate=ge_config[' lrb'],
            args=args
        )
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")
    network.set_seed(args.seed)

    # Training loop variables
    best_acc, no_imp, step = 0.0, 0, 0
    last_accuracies = []
    start = time.time()

    # Main training loop
    while step < args.max_steps:
        train_sampler.set_epoch(step)
        for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}",
                          disable=(local_rank != 0)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, network, use_fp16, scaler)
            step += 1

            # Evaluation checkpoint
            if step % args.eval_steps == 0:
                stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)
                if local_rank == 0:
                    acc = evaluate(model, eval_loader, model.device)
                    elapsed = time.time() - start
                    line = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
                    print(line)
                    log_f.write(line + "\n")
                    log_f.flush()
                    metrics.append({"step": step, "accuracy": acc, "time": elapsed})
                    wandb.log({"accuracy": acc, "step": step, "time": elapsed})

                    # Update running average
                    last_accuracies.append(acc)
                    if len(last_accuracies) > 5:
                        last_accuracies.pop(0)
                    avg_acc = sum(last_accuracies) / len(last_accuracies)
                    print(f"Last {len(last_accuracies)} accuracies: {last_accuracies} | Running Avg: {avg_acc:.4f}")

                    # Stopping criteria
                    if len(last_accuracies) == 5 and avg_acc >= args.target_accuracy:
                        print(f"Running average reached target accuracy {args.target_accuracy}. Stopping.")
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
                        stop_signal.fill_(1)
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
                        with open(os.path.join(args.output_dir, "metrics.json"), "w") as mf:
                            json.dump(metrics, mf, indent=2)
                        wandb.finish()
                    log_f.close()
                    return

            if step >= args.max_steps:
                break

    # Final save
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as mf:
            json.dump(metrics, mf, indent=2)
        wandb.finish()
    log_f.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tensor-parallel train with outputs saved")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--loss_type", type=str, default="ber", choices=["ber", "g-e"],
                        help='Type of packet loss simulation: "ber" or "g-e"')
    parser.add_argument("--ge_config", type=str, default="default")
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--dataset", type=str, default="winogrande")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--loss_rate", type=float, default=0.001)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--target_accuracy", type=float, default=0.75)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_to_accuracy(args)
