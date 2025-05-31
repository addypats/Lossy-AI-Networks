# import os
# import time
# import json
# import torch
# import yaml
# import numpy as np
# import torch.distributed as dist
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import torch.optim as optim
# import wandb
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from torch.cuda.amp import autocast, GradScaler
# from sklearn.metrics import accuracy_score

# from comms import LossyNetwork
# from parallel_layers_mod import replace_linears
# from data import get_dataset

# def train_step(model, inputs, optimizer, network, use_fp16, scaler, log_f, step):
#     model.train()
#     if use_fp16:
#         with autocast():
#             outputs = model(**inputs)
#             loss = outputs.loss
#         scaler.scale(loss).backward()
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 param.grad = network.receive(param.grad, network.send(param.grad))
#                 norm_val = param.grad.norm().item()
#                 log_f.write(f"Step {step} | FP16 | Grad Norm [{name}]: {norm_val:.6f}\n")
#                 if dist.get_rank() == 0:
#                     wandb.log({f"grad_norm/{name}": norm_val}, step=step)
#         scaler.step(optimizer)
#         scaler.update()
#     else:
#         outputs = model(**inputs)
#         loss = outputs.loss
#         loss.backward()
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 param.grad = network.receive(param.grad, network.send(param.grad))
#                 norm_val = param.grad.norm().item()
#                 log_f.write(f"Step {step} | FP32 | Grad Norm [{name}]: {norm_val:.6f}\n")
#                 if dist.get_rank() == 0:
#                     wandb.log({f"grad_norm/{name}": norm_val}, step=step)
#         optimizer.step()
#     optimizer.zero_grad()
#     if dist.get_rank() == 0:
#         wandb.log({"train/loss": loss.item()}, step=step)
#     return loss.item()

# def evaluate(model, eval_loader, device):
#     model.eval()
#     local_correct = torch.tensor(0, device=device, dtype=torch.long)
#     local_total   = torch.tensor(0, device=device, dtype=torch.long)

#     with torch.no_grad():
#         for batch in eval_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             logits = model(**batch).logits
#             preds  = torch.argmax(logits, dim=-1)
#             labels = batch['labels']
#             local_correct += (preds == labels).sum()
#             local_total   += labels.size(0)

#     dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
#     dist.all_reduce(local_total,   op=dist.ReduceOp.SUM)

#     return (local_correct.float() / local_total.float()).item()

# def train_to_accuracy(args):
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend='nccl', init_method='env://')
#     world_size = args.tensor_parallel_size
#     group = dist.group.WORLD

#     with open("src/config.yaml") as cf:
#         dataset_config = yaml.safe_load(cf)[args.dataset]

#     num_labels     = dataset_config['num_labels']
#     args.target_accuracy = dataset_config['target_acc']
#     report_ttac    = dataset_config.get('report_ttac', [])

#     os.makedirs(args.output_dir, exist_ok=True)
#     with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
#         yaml.dump(vars(args), f)
#     open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
#     log_file = os.path.join(args.output_dir, 'training.log')
#     log_f = open(log_file, 'w')
#     metrics = []

#     if local_rank == 0:
#         wandb.init(project="lossy-llama3-finetune", config=vars(args), name=args.run_name if hasattr(args, 'run_name') else None)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         args.model_name, num_labels=num_labels, token=args.token if hasattr(args, 'token') else True
#     )
#     model.gradient_checkpointing_disable()

#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name, token=args.token if hasattr(args, 'token') else True
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({"pad_token": "<pad>"})
#         model.resize_token_embeddings(len(tokenizer))
#     model.config.pad_token_id = tokenizer.pad_token_id

#     backbone = getattr(model, 'model', getattr(model, 'transformer', model))
#     replace_linears(backbone, world_size, group)
#     model.to(torch.cuda.current_device())

#     train_ds, eval_ds = get_dataset(args, tokenizer)
#     train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
#     eval_ds.set_format(type='torch',   columns=['input_ids','attention_mask','labels'])

#     train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
#                                        rank=local_rank, shuffle=True)
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
#     eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
#                                       rank=local_rank, shuffle=False)
#     eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

#     use_fp16 = getattr(args, "fp16", False)
#     scaler = GradScaler() if use_fp16 else None

#     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
#                             weight_decay=getattr(args, 'weight_decay', 0.0))
#     network = LossyNetwork(loss_rate=args.loss_rate)
#     network.set_seed(args.seed)

#     best_acc, no_imp, step = 0.0, 0, 0
#     start = time.time()

#     while step < args.max_steps:
#         train_sampler.set_epoch(step)
#         for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}", disable=(local_rank != 0)):
#             batch = {k: v.to(model.device) for k, v in batch.items()}
#             loss = train_step(model, batch, optimizer, network, use_fp16, scaler, log_f, step)

#             step += 1
#             if step % args.eval_steps == 0:
#                 acc = evaluate(model, eval_loader, model.device)
#                 stop_signal = torch.tensor(0, dtype=torch.uint8, device=model.device)

#                 if local_rank == 0:
#                     elapsed = time.time() - start
#                     print(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s")
#                     log_f.write(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s\n")
#                     log_f.flush()
#                     if dist.get_rank() == 0:
#                         wandb.log({"eval/accuracy": acc, "time": elapsed}, step=step)
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

#                 dist.broadcast(stop_signal, src=0, group=group)
#                 if stop_signal.item() == 1:
#                     if local_rank == 0:
#                         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
#                             json.dump(metrics, mf, indent=2)
#                         wandb.finish()
#                     log_f.close()
#                     return

#             if step >= args.max_steps:
#                 break

#     if local_rank == 0:
#         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
#         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
#             json.dump(metrics, mf, indent=2)
#         wandb.finish()
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


# import os
# import time
# import json
# import torch
# import yaml
# import numpy as np
# import torch.distributed as dist
# import torch.nn as nn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import wandb
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from torch.cuda.amp import autocast, GradScaler
# from sklearn.metrics import accuracy_score

# from comms import LossyNetwork
# from parallel_layers_mod import replace_linears
# from data import get_dataset

# def train_step(model, inputs, optimizer, network, use_fp16, scaler, log_f, step):
#     model.train()
#     if use_fp16:
#         with autocast():
#             outputs = model(**inputs)
#             loss = outputs.loss
#         scaler.scale(loss).backward()
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 param.grad = network.receive(param.grad, network.send(param.grad))
#                 norm_val = param.grad.norm().item()
#                 log_f.write(f"Step {step} | FP16 | Grad Norm [{name}]: {norm_val:.6f}\n")
#                 if dist.get_rank() == 0:
#                     wandb.log({f"grad_norm/{name}": norm_val}, step=step)
#         scaler.step(optimizer)
#         scaler.update()
#     else:
#         outputs = model(**inputs)
#         loss = outputs.loss
#         loss.backward()
#         for name, param in model.named_parameters():
#             if param.grad is not None:
#                 param.grad = network.receive(param.grad, network.send(param.grad))
#                 norm_val = param.grad.norm().item()
#                 log_f.write(f"Step {step} | FP32 | Grad Norm [{name}]: {norm_val:.6f}\n")
#                 if dist.get_rank() == 0:
#                     wandb.log({f"grad_norm/{name}": norm_val}, step=step)
#         optimizer.step()
#     optimizer.zero_grad()
#     if dist.get_rank() == 0:
#         wandb.log({"train/loss": loss.item()}, step=step)
#     return loss.item()

# def evaluate(model, eval_loader, device):
#     model.eval()
#     local_correct = torch.tensor(0, device=device, dtype=torch.long)
#     local_total   = torch.tensor(0, device=device, dtype=torch.long)

#     with torch.no_grad():
#         for batch in eval_loader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             logits = model(**batch).logits
#             preds  = torch.argmax(logits, dim=-1)
#             labels = batch['labels']
#             local_correct += (preds == labels).sum()
#             local_total   += labels.size(0)

#     dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
#     dist.all_reduce(local_total,   op=dist.ReduceOp.SUM)

#     return (local_correct.float() / local_total.float()).item()

# def train_to_accuracy(args):
#     local_rank = int(os.environ.get("LOCAL_RANK", 0))
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend='nccl', init_method='env://')
#     world_size = args.tensor_parallel_size
#     group = dist.group.WORLD

#     with open("src/config.yaml") as cf:
#         dataset_config = yaml.safe_load(cf)[args.dataset]

#     num_labels     = dataset_config['num_labels']
#     args.target_accuracy = dataset_config['target_acc']
#     report_ttac    = dataset_config.get('report_ttac', [])

#     os.makedirs(args.output_dir, exist_ok=True)
#     with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
#         yaml.dump(vars(args), f)
#     open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
#     log_file = os.path.join(args.output_dir, 'training.log')
#     log_f = open(log_file, 'w')
#     metrics = []

#     if local_rank == 0:
#         wandb.init(project="lossy-llama3-finetune", config=vars(args), name=args.run_name if hasattr(args, 'run_name') else None)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         args.model_name, num_labels=num_labels, token=args.token if hasattr(args, 'token') else True
#     )
#     model.gradient_checkpointing_disable()

#     tokenizer = AutoTokenizer.from_pretrained(
#         args.model_name, token=args.token if hasattr(args, 'token') else True
#     )
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({"pad_token": "<pad>"})
#         model.resize_token_embeddings(len(tokenizer))
#     model.config.pad_token_id = tokenizer.pad_token_id

#     backbone = getattr(model, 'model', getattr(model, 'transformer', model))
#     replace_linears(backbone, world_size, group)
#     model.to(torch.cuda.current_device())

#     train_ds, eval_ds = get_dataset(args, tokenizer)
#     train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
#     eval_ds.set_format(type='torch',   columns=['input_ids','attention_mask','labels'])

#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#     eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

#     use_fp16 = getattr(args, "fp16", False)
#     scaler = GradScaler() if use_fp16 else None

#     optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
#                             weight_decay=getattr(args, 'weight_decay', 0.0))
#     network = LossyNetwork(loss_rate=args.loss_rate)
#     network.set_seed(args.seed)

#     best_acc, no_imp, step = 0.0, 0, 0
#     start = time.time()

#     while step < args.max_steps:
#         for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}", disable=(local_rank != 0)):
#             batch = {k: v.to(model.device) for k, v in batch.items()}
#             loss = train_step(model, batch, optimizer, network, use_fp16, scaler, log_f, step)

#             step += 1
#             if step % args.eval_steps == 0:
#                 acc = evaluate(model, eval_loader, model.device)
#                 stop_signal = torch.tensor(0, dtype=torch.uint8, device=model.device)

#                 if local_rank == 0:
#                     elapsed = time.time() - start
#                     print(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s")
#                     log_f.write(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s\n")
#                     log_f.flush()
#                     if dist.get_rank() == 0:
#                         wandb.log({"eval/accuracy": acc, "time": elapsed}, step=step)
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

#                 dist.broadcast(stop_signal, src=0, group=group)
#                 if stop_signal.item() == 1:
#                     if local_rank == 0:
#                         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
#                             json.dump(metrics, mf, indent=2)
#                         wandb.finish()
#                     log_f.close()
#                     return

#             if step >= args.max_steps:
#                 break

#     if local_rank == 0:
#         torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
#         with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
#             json.dump(metrics, mf, indent=2)
#         wandb.finish()
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
#     with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
#         yaml.dump(vars(args), f)
#     open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()

#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     train_to_accuracy(args)




import os
import time
import json
import torch
import yaml
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaForSequenceClassification
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

from comms import LossyNetwork
from data import get_dataset
from tp_Llama_modules import TensorParallelLlamaModel
from tp_llama_sequence_classification import TPLlamaForSequenceClassification

def train_step(model, inputs, optimizer, network, use_fp16, scaler, log_f, step):
    model.train()
    if use_fp16:
        with autocast():
            outputs = model(**inputs)
            loss = outputs.loss
        scaler.scale(loss).backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad = network.receive(param.grad, network.send(param.grad))
                norm_val = param.grad.norm().item()
                log_f.write(f"Step {step} | FP16 | Grad Norm [{name}]: {norm_val:.6f}\n")
                if dist.get_rank() == 0:
                    wandb.log({f"grad_norm/{name}": norm_val}, step=step)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad = network.receive(param.grad, network.send(param.grad))
                norm_val = param.grad.norm().item()
                log_f.write(f"Step {step} | FP32 | Grad Norm [{name}]: {norm_val:.6f}\n")
                if dist.get_rank() == 0:
                    wandb.log({f"grad_norm/{name}": norm_val}, step=step)
        optimizer.step()
    optimizer.zero_grad()
    if dist.get_rank() == 0:
        wandb.log({"train/loss": loss.item()}, step=step)
    return loss.item()

def evaluate(model, eval_loader, device):
    model.eval()
    local_correct = torch.tensor(0, device=device, dtype=torch.long)
    local_total   = torch.tensor(0, device=device, dtype=torch.long)
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1)
            labels = batch['labels']
            local_correct += (preds == labels).sum()
            local_total += labels.size(0)
    dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_total, op=dist.ReduceOp.SUM)
    return (local_correct.float() / local_total.float()).item()

def train_to_accuracy(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = args.tensor_parallel_size
    group = dist.group.WORLD

    with open("src/config.yaml") as cf:
        dataset_config = yaml.safe_load(cf)[args.dataset]

    num_labels = dataset_config['num_labels']
    args.target_accuracy = dataset_config['target_acc']
    report_ttac = dataset_config.get('report_ttac', [])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
    log_file = os.path.join(args.output_dir, 'training.log')
    log_f = open(log_file, 'w')
    metrics = []

    if local_rank == 0:
        wandb.init(project="lossy-llama3-finetune", config=vars(args), name=args.run_name if hasattr(args, 'run_name') else None)

    if 'llama' in args.model_name.lower():
        from transformers import LlamaConfig
        config = LlamaConfig.from_pretrained(args.model_name)
        config.num_labels = num_labels
        # model = TensorParallelLlamaModel(config, world_size, group)
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        model = TPLlamaForSequenceClassification(config, world_size, group).to(device)
        # model = LlamaForSequenceClassification(config=config, model=model)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels, token=args.token if hasattr(args, 'token') else True
        )

    model.to(torch.cuda.current_device())

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=args.token if hasattr(args, 'token') else True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    train_ds, eval_ds = get_dataset(args, tokenizer)
    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

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
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    best_acc, no_imp, step = 0.0, 0, 0
    start = time.time()

    while step < args.max_steps:
        train_sampler.set_epoch(step)
        for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}", disable=(local_rank != 0)):
            
            # batch = {k: v.to(model.device) for k, v in batch.items()}
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            loss = train_step(model, batch, optimizer, network, use_fp16, scaler, log_f, step)

            step += 1
            if step % args.eval_steps == 0:
                # acc = evaluate(model, eval_loader, model.device)
                device = next(model.parameters()).device
                acc = evaluate(model, eval_loader, device)

                stop_signal = torch.tensor(0, dtype=torch.uint8, device=device)

                if local_rank == 0:
                    elapsed = time.time() - start
                    print(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s")
                    log_f.write(f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s\n")
                    log_f.flush()
                    wandb.log({"eval/accuracy": acc, "time": elapsed}, step=step)
                    metrics.append({'step': step, 'accuracy': acc, 'time': elapsed})

                    if acc >= args.target_accuracy:
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
                        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
                            json.dump(metrics, mf, indent=2)
                        wandb.finish()
                    log_f.close()
                    return

            if step >= args.max_steps:
                break

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
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--dataset', type=str, default='winogrande')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss_rate', type=float, default=0.001)
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--target_accuracy', type=float, default=0.75)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_to_accuracy(args)
