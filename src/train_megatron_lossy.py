#!/usr/bin/env python
"""
Clean Megatron-LM + Lossy Network Training Script

This script integrates Megatron-LM's tensor parallelism with your existing
lossy network research. It's designed to work with your existing bash script
and parameter-passing workflow.

Key features:
- Uses Megatron-LM for efficient tensor parallelism when available
- Falls back to your custom tensor parallel implementation if Megatron is not available
- Integrates your custom Lossy and Gilbert-Elliott network classes
- Compatible with your existing bash script and configuration workflow
- Maintains all your existing experimental parameters and logging
"""

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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import wandb

# Import your existing classes
from comms import LossyNetwork, GillbertElliotLossyNetwork
from data import get_dataset

# Try to import Megatron-LM, fall back to custom implementation if not available
try:
    from megatron.core.tensor_parallel import ColumnParallelLinear as MegatronColumnParallelLinear
    from megatron.core.tensor_parallel import RowParallelLinear as MegatronRowParallelLinear
    from megatron.core import parallel_state
    MEGATRON_AVAILABLE = True
    print("✅ Megatron-LM tensor parallelism available")
except ImportError:
    MEGATRON_AVAILABLE = False
    print("⚠️ Megatron-LM not available, using custom tensor parallelism")
    from parallel_layers_gpt import RowParallelLinear, ColumnParallelLinear
    from transformers.models.gpt2.modeling_gpt2 import Conv1D


def setup_megatron_if_available(tensor_parallel_size):
    """Initialize Megatron parallel state if available."""
    if MEGATRON_AVAILABLE:
        try:
            # Initialize Megatron's parallel state
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=tensor_parallel_size,
                pipeline_model_parallel_size=1
            )
            print(f"✅ Megatron parallel state initialized with TP size {tensor_parallel_size}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to initialize Megatron parallel state: {e}")
            return False
    return False


def inspect_model_structure(module, prefix="", max_depth=3, current_depth=0):
    """Inspect model structure to see what layers exist."""
    if current_depth >= max_depth:
        return
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            print(f"   📐 {full_name}: nn.Linear({child.in_features}, {child.out_features})")
        elif hasattr(child, 'weight') and len(child.weight.shape) == 2:
            weight_shape = tuple(child.weight.shape)
            layer_type = type(child).__name__
            if layer_type == 'Conv1D':
                print(f"   � {full_name}: {layer_type}{weight_shape} [GPT2 linear layer]")
            else:
                print(f"   📐 {full_name}: {layer_type}{weight_shape}")
        else:
            print(f"   📦 {full_name}: {type(child).__name__}")
            if current_depth < max_depth - 1:
                inspect_model_structure(child, full_name, max_depth, current_depth + 1)


def replace_linears_megatron(module, world_size, path=""):
    """Replace linear layers with Megatron tensor parallel versions."""
    replacement_count = 0
    
    for name, child in list(module.named_children()):
        current_path = f"{path}.{name}" if path else name
        
        if isinstance(child, nn.Linear):
            in_features = child.in_features
            out_features = child.out_features
            print(f"🔍 Found nn.Linear at {current_path}: {in_features} -> {out_features}")
            
            # Use row-parallel for layers where output can be sharded
            if out_features % world_size == 0:
                print(f"✅ Output dimension {out_features} is divisible by world_size {world_size}, creating RowParallel...")
                # Create Megatron row-parallel linear
                device = child.weight.device
                dtype = child.weight.dtype
                
                # Megatron expects input_size, output_size
                new_layer = MegatronRowParallelLinear(
                    input_size=in_features,
                    output_size=out_features,
                    bias=child.bias is not None,
                    input_is_parallel=False,
                    skip_bias_add=False,
                    device=device,
                    dtype=dtype
                )
                
                # Copy weights - Megatron will handle sharding
                with torch.no_grad():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    output_size_per_partition = out_features // world_size
                    start_idx = rank * output_size_per_partition
                    end_idx = start_idx + output_size_per_partition
                    
                    new_layer.weight.copy_(child.weight[start_idx:end_idx])
                    if child.bias is not None:
                        new_layer.bias.copy_(child.bias[start_idx:end_idx])
                
                setattr(module, name, new_layer)
                replacement_count += 1
                print(f"✅ Replaced {current_path} with Megatron RowParallelLinear: {out_features} -> {output_size_per_partition}")
            
            elif in_features % world_size == 0:
                print(f"✅ Input dimension {in_features} is divisible by world_size {world_size}, creating ColumnParallel...")
                # Create Megatron column-parallel linear for input sharding
                device = child.weight.device
                dtype = child.weight.dtype
                
                new_layer = MegatronColumnParallelLinear(
                    input_size=in_features,
                    output_size=out_features,
                    bias=child.bias is not None,
                    gather_output=True,
                    device=device,
                    dtype=dtype
                )
                
                # Copy weights - Megatron will handle sharding
                with torch.no_grad():
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    input_size_per_partition = in_features // world_size
                    start_idx = rank * input_size_per_partition
                    end_idx = start_idx + input_size_per_partition
                    
                    new_layer.weight.copy_(child.weight[:, start_idx:end_idx])
                    if child.bias is not None and rank == 0:
                        new_layer.bias.copy_(child.bias)
                
                setattr(module, name, new_layer)
                replacement_count += 1
                print(f"✅ Replaced {current_path} with Megatron ColumnParallelLinear: {in_features} -> {input_size_per_partition}")
            else:
                print(f"⚠️ Skipped {current_path}: neither input ({in_features}) nor output ({out_features}) divisible by world_size {world_size}")
        else:
            # Recurse into child modules
            child_replacements = replace_linears_megatron(child, world_size, current_path)
            replacement_count += child_replacements
    
    return replacement_count


def replace_linears_custom(module, world_size, group):
    """Replace linear layers with custom tensor parallel versions (fallback)."""
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Linear, Conv1D)):
            # Handle both nn.Linear and Conv1D
            if isinstance(child, Conv1D):
                # Conv1D.weight is [in_features, out_features]
                in_f, out_f = child.weight.shape
            else:
                # nn.Linear.weight is [out_features, in_features]
                out_f, in_f = child.weight.shape

            # Use row-parallel for consistency
            if out_f % world_size == 0:
                wrapped = RowParallelLinear(child, world_size, group)
                setattr(module, name, wrapped)
                print(f"✅ Replaced {name} with custom RowParallelLinear: [{out_f}, {in_f}]")
            else:
                print(f"⚠️ Skipped {name}: output dim {out_f} not divisible by world_size {world_size}")
        else:
            replace_linears_custom(child, world_size, group)


def train_step(model, inputs, optimizer, network, use_fp16=False, scaler=None):
    """Single training step with lossy communication simulation."""
    model.train()
    
    if use_fp16 and scaler is not None:
        with autocast():
            outputs = model(**inputs)
            loss = outputs.loss
        
        scaler.scale(loss).backward()
        
        # Apply lossy communication to gradients
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                mask = network.send(param.grad)
                param.grad = network.receive(param.grad, mask)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        
        # Apply lossy communication to gradients
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                mask = network.send(param.grad)
                param.grad = network.receive(param.grad, mask)
        
        optimizer.step()
    
    optimizer.zero_grad()
    return loss.item()


def evaluate(model, eval_loader, device):
    """Evaluate the model on the validation set."""
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


def setup_lossy_network(args):
    """Setup the appropriate lossy network based on configuration."""
    if args.loss_type == 'g-e':
        # Load Gilbert-Elliott parameters from CSV
        import pandas as pd
        try:
            ge_params = pd.read_csv('g_e_params.csv')
            config_row = ge_params[ge_params['config_id'] == args.ge_config]
            if config_row.empty:
                raise ValueError(f"G-E config '{args.ge_config}' not found in g_e_params.csv")
            
            p_gb = float(config_row['p_gb'].iloc[0])
            p_bg = float(config_row['p_bg'].iloc[0])
            good_loss_rate = float(config_row['good_loss_rate'].iloc[0])
            bad_loss_rate = float(config_row['bad_loss_rate'].iloc[0])
            
            network = GillbertElliotLossyNetwork(p_gb, p_bg, good_loss_rate, bad_loss_rate, args)
            print(f"✅ Using Gilbert-Elliott network: config={args.ge_config}, p_gb={p_gb}, p_bg={p_bg}")
            
        except Exception as e:
            print(f"⚠️ Failed to load G-E config: {e}, falling back to Bernoulli")
            network = LossyNetwork(args)
    else:
        # Default Bernoulli (uniform random) loss
        network = LossyNetwork(args)
        print(f"✅ Using Bernoulli loss network: loss_rate={args.loss_rate}")
    
    network.set_seed(args.seed)
    return network


def train_to_accuracy(args):
    """Main training function."""
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://')
    dist.init_process_group(backend='gloo', init_method='env://')
    
    world_size = dist.get_world_size()
    if world_size != args.tensor_parallel_size:
        print(f"⚠️ Warning: actual world_size ({world_size}) != tensor_parallel_size ({args.tensor_parallel_size})")
        args.tensor_parallel_size = world_size
    
    group = dist.group.WORLD
    
    # Setup Megatron if available
    use_megatron = setup_megatron_if_available(args.tensor_parallel_size)
    
    # Load dataset configuration
    with open("src/config.yaml") as cf:
        dataset_config = yaml.safe_load(cf)[args.dataset]
    
    num_labels = dataset_config['num_labels']
    args.target_accuracy = dataset_config['target_acc']
    
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, 'training.log')
    log_f = open(log_file, 'w')
    metrics = []
    
    # Initialize Weights & Biases
    if local_rank == 0:
        run_id = os.path.basename(os.path.abspath(args.output_dir))
        wandb.init(
            project="lossy-network-research",
            name=run_id,
            config=vars(args),
            tags=["megatron" if use_megatron else "custom", args.loss_type, args.dataset]
        )
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, use_auth_token=True
    )
    model.gradient_checkpointing_disable()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
    
    # Setup tensor parallelism
    if hasattr(model, 'model'):
        backbone = model.model
        backbone_name = "model.model"
    elif hasattr(model, 'transformer'):
        backbone = model.transformer
        backbone_name = "model.transformer"
    else:
        backbone = model
        backbone_name = "model"
    
    print(f"🏗️ Model structure analysis:")
    print(f"   - Backbone: {backbone_name}")
    print(f"   - Backbone type: {type(backbone)}")
    print(f"   - World size: {args.tensor_parallel_size}")
    print(f"   - Using Megatron: {use_megatron}")
    
    print(f"\n🔍 Model layer inspection:")
    inspect_model_structure(backbone)
    
    if use_megatron:
        print(f"\n🔧 Starting Megatron layer replacement...")
        replacements = replace_linears_megatron(backbone, args.tensor_parallel_size)
        print(f"✅ Megatron replacement complete: {replacements} layers replaced")
    else:
        print(f"\n🔧 Starting custom layer replacement...")
        replace_linears_custom(backbone, args.tensor_parallel_size, group)
        print(f"✅ Custom replacement complete")
    
    model.to(torch.cuda.current_device())
    
    # Setup datasets
    train_ds, eval_ds = get_dataset(args, tokenizer)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Setup data loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=local_rank, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)
    
    # Setup training components
    use_fp16 = getattr(args, "fp16", False)
    scaler = GradScaler() if use_fp16 else None
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=getattr(args, 'weight_decay', 0.0)
    )
    
    # Setup lossy network
    network = setup_lossy_network(args)
    
    # Training loop
    best_acc = 0.0
    no_improvement = 0
    step = 0
    start_time = time.time()
    last_accuracies = []
    
    print(f"🚀 Starting training with {'Megatron-LM' if use_megatron else 'custom'} tensor parallelism")
    print(f"📊 Dataset: {args.dataset}, Model: {args.model_name}")
    print(f"🔀 Loss type: {args.loss_type}, Loss rate: {args.loss_rate}")
    
    while step < args.max_steps:
        train_sampler.set_epoch(step)
        
        for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}", disable=(local_rank != 0)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Training step with lossy communication
            loss = train_step(model, batch, optimizer, network, use_fp16, scaler)
            step += 1
            
            # Evaluation and logging
            if step % args.eval_steps == 0:
                stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)
                
                if local_rank == 0:
                    acc = evaluate(model, eval_loader, model.device)
                    elapsed = time.time() - start_time
                    line = f"Step {step} | Acc {acc:.4f} | Loss {loss:.4f} | Time {elapsed:.1f}s"
                    print(line)
                    log_f.write(line + "\n")
                    log_f.flush()
                    
                    # Log to wandb
                    wandb.log({
                        'accuracy': acc,
                        'loss': loss,
                        'step': step,
                        'time': elapsed,
                        'learning_rate': args.learning_rate
                    })
                    
                    metrics.append({
                        'step': step,
                        'accuracy': acc,
                        'loss': loss,
                        'time': elapsed
                    })
                    
                    # Track last 5 accuracies for stability
                    last_accuracies.append(acc)
                    if len(last_accuracies) > 5:
                        last_accuracies.pop(0)
                    
                    avg_acc = sum(last_accuracies) / len(last_accuracies)
                    print(f"Last {len(last_accuracies)} accuracies: {[f'{a:.4f}' for a in last_accuracies]} | Avg: {avg_acc:.4f}")
                    
                    # Check stopping conditions
                    if len(last_accuracies) == 5 and avg_acc >= args.target_accuracy:
                        print(f"🎯 Target accuracy reached! Running average: {avg_acc:.4f}")
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
                        stop_signal.fill_(1)
                    elif acc > best_acc:
                        best_acc = acc
                        no_improvement = 0
                        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                    else:
                        no_improvement += 1
                        if no_improvement >= args.patience:
                            print(f"⏰ Early stopping: no improvement for {args.patience} evaluations")
                            stop_signal.fill_(1)
                
                # Broadcast stop signal to all ranks
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
    
    # Final cleanup
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
            json.dump(metrics, mf, indent=2)
        wandb.finish()
    
    log_f.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Megatron-LM + Lossy Network Training")
    
    # Core training parameters
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--tensor_parallel_size', type=int, default=4, help='Tensor parallel world size')
    parser.add_argument('--model_name', type=str, default='gpt2-large', help='Model name or path')
    parser.add_argument('--dataset', type=str, default='winogrande', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    
    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    
    # Lossy network parameters
    parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e'],
                        help='Loss type: "ber" for Bernoulli, "g-e" for Gilbert-Elliott')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Loss rate for Bernoulli model')
    parser.add_argument('--ge_config', type=str, default='default', 
                        help='Gilbert-Elliott config ID (from g_e_params.csv)')
    
    # Training control
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--max_samples', type=int, default=0, help='Max samples (0 = use all)')
    parser.add_argument('--target_accuracy', type=float, default=0.75, help='Target accuracy for early stopping')
    parser.add_argument('--eval_steps', type=int, default=100, help='Steps between evaluations')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--max_steps', type=int, default=100000, help='Maximum training steps')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup output directory and save arguments
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create compatibility file for your existing workflow
    open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
    
    # Start training
    train_to_accuracy(args)
