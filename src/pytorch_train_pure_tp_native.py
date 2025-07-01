#!/usr/bin/env python
"""
Pure Tensor Parallelism implementation (Megatron-LM style) with lossy networks
No external dependencies - uses only PyTorch primitives
"""
import os
import time
import json
import torch
import yaml
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import wandb

from comms import LossyNetwork, GillbertElliotLossyNetwork
from data import get_dataset


class TensorParallelLinear(nn.Module):
    """
    Pure tensor parallel linear layer implementation
    """
    def __init__(self, in_features, out_features, parallel_dim='output', 
                 lossy_network=None, bias=True, gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.parallel_dim = parallel_dim
        self.lossy_network = lossy_network
        self.gather_output = gather_output
        
        # Get TP world size and rank
        self.tp_world_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        
        if parallel_dim == 'output':
            # Column parallel: split output dimension across ranks
            assert out_features % self.tp_world_size == 0, \
                f"out_features {out_features} not divisible by tp_world_size {self.tp_world_size}"
            self.output_size_per_partition = out_features // self.tp_world_size
            self.weight = nn.Parameter(torch.randn(self.output_size_per_partition, in_features))
            self.bias = nn.Parameter(torch.randn(self.output_size_per_partition)) if bias else None
            
        elif parallel_dim == 'input':
            # Row parallel: split input dimension across ranks
            assert in_features % self.tp_world_size == 0, \
                f"in_features {in_features} not divisible by tp_world_size {self.tp_world_size}"
            self.input_size_per_partition = in_features // self.tp_world_size
            self.weight = nn.Parameter(torch.randn(out_features, self.input_size_per_partition))
            self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
            
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def _apply_lossy_communication(self, tensor, operation_type="all_gather"):
        """Apply lossy network simulation to communication"""
        if self.lossy_network is None:
            return tensor
            
        # Apply loss simulation
        mask = self.lossy_network.send(tensor)
        return self.lossy_network.receive(tensor, mask)
        
    def forward(self, input_):
        if self.parallel_dim == 'output':
            # Column parallel: f(XA) = [XA_1, XA_2, ..., XA_p]
            output_parallel = torch.matmul(input_, self.weight.t())
            if self.bias is not None:
                output_parallel = output_parallel + self.bias
                
            if self.gather_output:
                # Apply lossy communication before all-gather
                output_parallel = self._apply_lossy_communication(output_parallel, "all_gather")
                
                # All-gather outputs from all ranks
                output_list = [torch.zeros_like(output_parallel) for _ in range(self.tp_world_size)]
                dist.all_gather(output_list, output_parallel)
                output = torch.cat(output_list, dim=-1)
                return output
            else:
                return output_parallel
                
        elif self.parallel_dim == 'input':
            # Row parallel: f(XA) = f([X_1, X_2, ..., X_p] * [A_1; A_2; ...; A_p])
            # Each rank computes X_i * A_i, then we sum via all-reduce
            
            # Apply lossy communication to input before computation
            input_lossy = self._apply_lossy_communication(input_, "reduce_scatter")
            
            # Compute partial result: X_i * A_i
            output_parallel = torch.matmul(input_lossy, self.weight.t())
            
            # All-reduce to sum partial results across ranks
            dist.all_reduce(output_parallel, op=dist.ReduceOp.SUM)
            
            # Add bias only once (on rank 0)
            if self.bias is not None and self.tp_rank == 0:
                output_parallel = output_parallel + self.bias
                
            return output_parallel


class TensorParallelEmbedding(nn.Module):
    """Tensor parallel embedding layer"""
    def __init__(self, num_embeddings, embedding_dim, lossy_network=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.lossy_network = lossy_network
        
        self.tp_world_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()
        
        # Split embedding dimension
        assert embedding_dim % self.tp_world_size == 0
        self.embedding_dim_per_partition = embedding_dim // self.tp_world_size
        
        self.weight = nn.Parameter(torch.randn(num_embeddings, self.embedding_dim_per_partition))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
    def forward(self, input_ids):
        # Each rank computes its partition of embeddings
        embeddings_parallel = nn.functional.embedding(input_ids, self.weight)
        
        # Apply lossy communication before all-gather
        if self.lossy_network is not None:
            mask = self.lossy_network.send(embeddings_parallel)
            embeddings_parallel = self.lossy_network.receive(embeddings_parallel, mask)
        
        # All-gather to get full embeddings
        embeddings_list = [torch.zeros_like(embeddings_parallel) for _ in range(self.tp_world_size)]
        dist.all_gather(embeddings_list, embeddings_parallel)
        embeddings = torch.cat(embeddings_list, dim=-1)
        
        return embeddings


def replace_with_pure_tensor_parallel(model, lossy_network):
    """
    Replace model layers with pure tensor parallel versions
    """
    tp_world_size = dist.get_world_size()
    replaced_count = 0
    
    def _replace_module(module, name=""):
        nonlocal replaced_count
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                in_features = child.in_features
                out_features = child.out_features
                
                # Strategy: use column parallel for most layers, row parallel for output layers
                if "score" in child_name.lower() or "classifier" in child_name.lower():
                    # Output layers: use row parallel (don't gather output)
                    if in_features % tp_world_size == 0:
                        new_layer = TensorParallelLinear(
                            in_features, out_features,
                            parallel_dim='input',
                            lossy_network=lossy_network,
                            bias=child.bias is not None,
                            gather_output=False  # Don't gather for final output
                        )
                        print(f"Replaced {name}.{child_name} with row-parallel TP layer (output)")
                    else:
                        print(f"Skipping {name}.{child_name}: input dim not divisible")
                        _replace_module(child, f"{name}.{child_name}")
                        continue
                        
                elif out_features % tp_world_size == 0:
                    # Hidden layers: use column parallel
                    new_layer = TensorParallelLinear(
                        in_features, out_features,
                        parallel_dim='output',
                        lossy_network=lossy_network,
                        bias=child.bias is not None,
                        gather_output=True
                    )
                    print(f"Replaced {name}.{child_name} with column-parallel TP layer")
                    
                elif in_features % tp_world_size == 0:
                    # Use row parallel as fallback
                    new_layer = TensorParallelLinear(
                        in_features, out_features,
                        parallel_dim='input',
                        lossy_network=lossy_network,
                        bias=child.bias is not None,
                        gather_output=True
                    )
                    print(f"Replaced {name}.{child_name} with row-parallel TP layer")
                    
                else:
                    print(f"Skipping {name}.{child_name}: dimensions not divisible by {tp_world_size}")
                    _replace_module(child, f"{name}.{child_name}")
                    continue
                
                # Copy weights to the new layer
                _copy_weights(child, new_layer)
                setattr(module, child_name, new_layer)
                replaced_count += 1
                
            elif isinstance(child, nn.Embedding):
                # Replace embeddings with tensor parallel version
                if child.embedding_dim % tp_world_size == 0:
                    new_embedding = TensorParallelEmbedding(
                        child.num_embeddings, child.embedding_dim,
                        lossy_network=lossy_network
                    )
                    _copy_embedding_weights(child, new_embedding)
                    setattr(module, child_name, new_embedding)
                    replaced_count += 1
                    print(f"Replaced {name}.{child_name} with TP embedding")
                else:
                    print(f"Skipping embedding {name}.{child_name}: dim not divisible")
                    
            else:
                _replace_module(child, f"{name}.{child_name}")
    
    _replace_module(model, "model")
    print(f"Replaced {replaced_count} layers with pure tensor parallel versions")
    return model


def _copy_weights(original_layer, new_layer):
    """Copy weights from original layer to tensor parallel layer"""
    tp_rank = dist.get_rank()
    tp_world_size = dist.get_world_size()
    
    with torch.no_grad():
        if new_layer.parallel_dim == 'output':
            # Column parallel: split output dimension
            out_per_partition = new_layer.out_features // tp_world_size
            start_idx = tp_rank * out_per_partition
            end_idx = start_idx + out_per_partition
            
            new_layer.weight.copy_(original_layer.weight[start_idx:end_idx, :])
            if original_layer.bias is not None and new_layer.bias is not None:
                new_layer.bias.copy_(original_layer.bias[start_idx:end_idx])
                
        elif new_layer.parallel_dim == 'input':
            # Row parallel: split input dimension
            in_per_partition = new_layer.in_features // tp_world_size
            start_idx = tp_rank * in_per_partition
            end_idx = start_idx + in_per_partition
            
            new_layer.weight.copy_(original_layer.weight[:, start_idx:end_idx])
            if original_layer.bias is not None and new_layer.bias is not None:
                if tp_rank == 0:
                    new_layer.bias.copy_(original_layer.bias)
                else:
                    new_layer.bias.zero_()


def _copy_embedding_weights(original_embedding, new_embedding):
    """Copy weights from original embedding to tensor parallel embedding"""
    tp_rank = dist.get_rank()
    tp_world_size = dist.get_world_size()
    
    dim_per_partition = new_embedding.embedding_dim // tp_world_size
    start_idx = tp_rank * dim_per_partition
    end_idx = start_idx + dim_per_partition
    
    with torch.no_grad():
        new_embedding.weight.copy_(original_embedding.weight[:, start_idx:end_idx])


def evaluate(model, eval_loader, device, local_rank):
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


def train_pure_tp_native(args):
    """Training with pure tensor parallelism (native PyTorch implementation)"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Initialize distributed
    dist.init_process_group(backend='nccl', init_method='env://')
    world_size = dist.get_world_size()
    
    # Validate tensor parallel size
    if world_size != args.tensor_parallel_size:
        print(f"Warning: world_size ({world_size}) != tensor_parallel_size ({args.tensor_parallel_size})")
        args.tensor_parallel_size = world_size
    
    print(f"[Rank {local_rank}] Pure tensor parallelism initialized (native implementation)")
    print(f"[Rank {local_rank}] World size: {world_size}")
    
    # Load configuration
    with open("src/config.yaml") as cf:
        dataset_config = yaml.safe_load(cf)[args.dataset]

    num_labels = dataset_config['num_labels']
    args.target_accuracy = dataset_config['target_acc']

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    log_file = os.path.join(args.output_dir, 'training.log')
    log_f = open(log_file, 'w')
    metrics = []

    # wandb initialization (only on rank 0)
    if local_rank == 0:
        run_id = os.path.basename(os.path.abspath(args.output_dir))
        wandb.init(
            project="pure_tp_native_lossy_gpt",
            name=run_id,
            config=vars(args)
        )

    # Load model and tokenizer
    print(f"[Rank {local_rank}] Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, token=True
    )
    model.gradient_checkpointing_disable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    # Initialize lossy network
    if args.loss_type == 'ber':
        network = LossyNetwork(args)
    elif args.loss_type == 'g-e':
        import pandas as pd
        configs = pd.read_csv('g_e_params.csv')
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

    # Apply pure tensor parallelism
    print(f"[Rank {local_rank}] Applying pure tensor parallelism...")
    model = replace_with_pure_tensor_parallel(model, network)
    model.to(torch.cuda.current_device())

    # Data loading
    print(f"[Rank {local_rank}] Loading data...")
    start_data_time = time.time()
    
    train_ds, eval_ds = get_dataset(args, tokenizer)
    
    data_load_time = time.time() - start_data_time
    print(f"[Rank {local_rank}] Data loading completed in {data_load_time:.2f}s")
    print(f"[Rank {local_rank}] Train dataset size: {len(train_ds)}")
    
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # For pure TP, we still distribute data across ranks for efficiency
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
                                      rank=local_rank, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)

    # Training setup
    use_fp16 = getattr(args, "fp16", False)
    scaler = GradScaler() if use_fp16 else None

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=getattr(args, 'weight_decay', 0.0))

    best_acc, no_imp, step = 0.0, 0, 0
    last_accuracies = []
    start = time.time()

    print(f"[Rank {local_rank}] Starting training with pure tensor parallelism...")

    # Training loop
    while step < args.max_steps:
        train_sampler.set_epoch(step)
        for batch in train_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            model.train()
            if use_fp16:
                with autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            step += 1

            # Evaluation
            if step % args.eval_steps == 0:
                stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)

                if local_rank == 0:
                    acc = evaluate(model, eval_loader, model.device, local_rank)
                    elapsed = time.time() - start
                    line = f"Step {step} | Acc {acc:.4f} | Loss {loss:.4f} | Time {elapsed:.1f}s"
                    print(line)
                    log_f.write(line + "\n")
                    log_f.flush()
                    metrics.append({'step': step, 'accuracy': acc, 'loss': loss.item(), 'time': elapsed})
                    wandb.log({'accuracy': acc, 'loss': loss.item(), 'step': step, 'time': elapsed})
                    
                    # Check target accuracy
                    last_accuracies.append(acc)
                    if len(last_accuracies) > 5:
                        last_accuracies.pop(0)
                    
                    if len(last_accuracies) >= 3 and all(a >= args.target_accuracy for a in last_accuracies[-3:]):
                        print(f"Target accuracy {args.target_accuracy} achieved!")
                        stop_signal.fill_(1)
                    elif acc > best_acc:
                        best_acc, no_imp = acc, 0
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    else:
                        no_imp += 1
                        if no_imp >= args.patience:
                            stop_signal.fill_(1)

                dist.broadcast(stop_signal, src=0)
                if stop_signal.item() == 1:
                    if local_rank == 0:
                        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
                            json.dump(metrics, mf, indent=2)
                        wandb.finish()
                    log_f.close()
                    return

            if step >= args.max_steps:
                break

    # Final cleanup
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
            json.dump(metrics, mf, indent=2)
        wandb.finish()
    log_f.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pure tensor parallel training (native) with lossy networks")
    parser.add_argument('--max_length', type=int, default=256)
    
    # Loss simulation params
    parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e'])
    parser.add_argument('--ge_config', type=str, default='default')

    parser.add_argument('--tensor_parallel_size', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='gpt2-large')
    parser.add_argument('--dataset', type=str, default='mnli')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss_rate', type=float, default=0.001)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--target_accuracy', type=float, default=0.75)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--output_dir', type=str, default='./output')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_pure_tp_native(args)
