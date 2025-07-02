#!/usr/bin/env python3
"""
Complete Megatron-LM + Your Existing Lossy Network Integration
This script provides a complete integration using your existing training setup,
bash scripts, lossy network classes, and experimental configuration.

This is the RECOMMENDED approach for your research - it leverages Megatron-LM's
optimized tensor parallelism while preserving all your existing research infrastructure.
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
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import wandb
import pandas as pd

# Your existing imports
from comms import LossyNetwork, GillbertElliotLossyNetwork
from data import get_dataset

class MegatronLossyIntegration:
    """
    Integration layer that replaces your custom tensor parallelism with Megatron-LM
    while preserving all your existing lossy network logic and experimental setup.
    """
    
    def __init__(self, args, lossy_network):
        """
        Initialize the Megatron integration.
        
        Args:
            args: Your existing argument namespace
            lossy_network: Your existing LossyNetwork or GillbertElliotLossyNetwork instance
        """
        self.args = args
        self.lossy_network = lossy_network
        self.original_functions = {}
        self.is_enabled = False
        
        # Statistics tracking
        self.stats = {
            'total_operations': 0,
            'lossy_operations': 0,
            'forward_ops': 0,
            'backward_ops': 0,
        }
        
    def enable_lossy_communication(self):
        """Enable lossy communication by hooking into Megatron's tensor parallel operations."""
        if self.is_enabled:
            return
            
        try:
            # Try to import Megatron's tensor parallel mappings
            from megatron.core.tensor_parallel import mappings
            
            # Store original functions for restoration
            self.original_functions = {
                'reduce': mappings._reduce,
                'gather_last_dim': mappings._gather_along_last_dim,
                'split_last_dim': mappings._split_along_last_dim,
            }
            
            # Try to get sequence parallel functions (may not exist in all versions)
            try:
                self.original_functions['reduce_scatter_first_dim'] = mappings._reduce_scatter_along_first_dim
                self.original_functions['gather_first_dim'] = mappings._gather_along_first_dim
            except AttributeError:
                # Not all Megatron versions have sequence parallelism
                pass
            
            # Replace with lossy versions
            mappings._reduce = self._lossy_reduce
            mappings._gather_along_last_dim = self._lossy_gather_last_dim
            mappings._split_along_last_dim = self._lossy_split_last_dim
            
            # Replace sequence parallel functions if they exist
            if 'reduce_scatter_first_dim' in self.original_functions:
                mappings._reduce_scatter_along_first_dim = self._lossy_reduce_scatter_first_dim
            if 'gather_first_dim' in self.original_functions:
                mappings._gather_along_first_dim = self._lossy_gather_first_dim
            
            self.is_enabled = True
            print("✅ Megatron lossy communication integration enabled")
            
        except ImportError as e:
            print(f"⚠️ Megatron-LM not available, falling back to your custom tensor parallelism: {e}")
            print("   Install Megatron-LM for production-scale experiments:")
            print("   git clone https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM && pip install -e .")
            return False
        except Exception as e:
            print(f"⚠️ Error enabling Megatron integration: {e}")
            return False
            
        return True
    
    def disable_lossy_communication(self):
        """Restore original Megatron functions."""
        if not self.is_enabled:
            return
            
        try:
            from megatron.core.tensor_parallel import mappings
            
            # Restore all original functions
            mappings._reduce = self.original_functions['reduce']
            mappings._gather_along_last_dim = self.original_functions['gather_last_dim']
            mappings._split_along_last_dim = self.original_functions['split_last_dim']
            
            if 'reduce_scatter_first_dim' in self.original_functions:
                mappings._reduce_scatter_along_first_dim = self.original_functions['reduce_scatter_first_dim']
            if 'gather_first_dim' in self.original_functions:
                mappings._gather_along_first_dim = self.original_functions['gather_first_dim']
            
            self.is_enabled = False
            print("✅ Megatron lossy communication integration disabled")
            
        except Exception as e:
            print(f"⚠️ Error disabling Megatron integration: {e}")
    
    def _apply_lossy_logic(self, tensor, operation_name):
        """Apply your existing lossy network logic to tensors."""
        self.stats['total_operations'] += 1
        
        # Detect if this is a gradient tensor (backward pass)
        is_gradient = tensor.requires_grad and hasattr(tensor, 'grad_fn') and tensor.grad_fn is not None
        
        if is_gradient:
            self.stats['backward_ops'] += 1
        else:
            self.stats['forward_ops'] += 1
        
        try:
            # Use your existing lossy network's send/receive pattern
            mask = self.lossy_network.send(tensor)
            processed_tensor = self.lossy_network.receive(tensor, mask)
            
            # Check if any losses were actually applied
            if not torch.equal(tensor, processed_tensor):
                self.stats['lossy_operations'] += 1
            
            return processed_tensor
            
        except Exception as e:
            print(f"⚠️ Error applying lossy logic in {operation_name}: {e}")
            return tensor
    
    def _lossy_reduce(self, tensor, group=None):
        """Lossy version of tensor parallel all-reduce."""
        processed_tensor = self._apply_lossy_logic(tensor, "all_reduce")
        return self.original_functions['reduce'](processed_tensor, group)
    
    def _lossy_gather_last_dim(self, tensor, group=None):
        """Lossy version of tensor parallel gather along last dimension."""
        processed_tensor = self._apply_lossy_logic(tensor, "gather_last_dim")
        return self.original_functions['gather_last_dim'](processed_tensor, group)
    
    def _lossy_split_last_dim(self, tensor, group=None):
        """Lossy version of tensor parallel split along last dimension."""
        processed_tensor = self._apply_lossy_logic(tensor, "split_last_dim")
        return self.original_functions['split_last_dim'](processed_tensor, group)
    
    def _lossy_reduce_scatter_first_dim(self, tensor, group=None):
        """Lossy version of sequence parallel reduce-scatter."""
        processed_tensor = self._apply_lossy_logic(tensor, "reduce_scatter_first_dim")
        return self.original_functions['reduce_scatter_first_dim'](processed_tensor, group)
    
    def _lossy_gather_first_dim(self, tensor, group=None):
        """Lossy version of sequence parallel all-gather."""
        processed_tensor = self._apply_lossy_logic(tensor, "gather_first_dim")
        return self.original_functions['gather_first_dim'](processed_tensor, group)
    
    def get_stats(self):
        """Get statistics about lossy operations."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


def create_megatron_model(args, num_labels):
    """
    Create a model using Megatron-LM's GPT architecture.
    Falls back to your existing approach if Megatron is not available.
    """
    try:
        from megatron.core.models.gpt import GPTModel
        from megatron.core import ModelParallelConfig
        
        # Create Megatron model configuration
        config = ModelParallelConfig(
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=1,  # We're only using tensor parallelism
            sequence_parallel=True,  # Enable sequence parallelism for efficiency
        )
        
        # Model architecture configuration
        model_config = {
            'num_layers': 12,  # Adjust based on your model
            'hidden_size': 768,  # Adjust based on your model
            'num_attention_heads': 12,  # Adjust based on your model
            'seq_length': args.max_length,
            'max_position_embeddings': args.max_length,
            'vocab_size': 50257,  # GPT-2 vocab size, adjust as needed
        }
        
        # Create Megatron model
        model = GPTModel(
            config=config,
            transformer_layer_spec=None,  # Use default transformer spec
            vocab_size=model_config['vocab_size'],
            max_sequence_length=model_config['seq_length'],
        )
        
        # Add classification head for fine-tuning
        model.classifier = nn.Linear(model_config['hidden_size'], num_labels)
        
        print("✅ Using Megatron-LM GPT model with optimized tensor parallelism")
        return model, True  # True indicates this is a Megatron model
        
    except ImportError:
        print("⚠️ Megatron-LM not available, using your existing HuggingFace + custom TP approach")
        
        # Fall back to your existing approach
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels, use_auth_token=True
        )
        model.gradient_checkpointing_disable()
        
        return model, False  # False indicates this is not a Megatron model


def setup_lossy_network(args):
    """Initialize your existing lossy network based on configuration."""
    if args.loss_type == 'ber':
        network = LossyNetwork(args)
    elif args.loss_type == 'g-e':
        # Load Gilbert-Elliott parameters from your CSV file
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
    return network


def train_step_with_megatron(model, inputs, optimizer, megatron_integration, is_megatron_model):
    """
    Training step that works with both Megatron and your existing models.
    """
    model.train()
    
    if is_megatron_model:
        # Megatron model forward pass
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        
        # Forward pass through Megatron model
        outputs = model(input_ids, attention_mask)
        
        # Get logits and compute loss
        logits = model.classifier(outputs)
        labels = inputs['labels']
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        
    else:
        # Your existing HuggingFace model approach
        outputs = model(**inputs)
        loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Apply lossy logic to gradients (your existing approach)
    # Note: With Megatron integration, losses are applied during tensor parallel communication
    # But we can still apply additional gradient-level losses if needed
    if not megatron_integration.is_enabled:
        # Fallback to your existing gradient-level loss application
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                mask = megatron_integration.lossy_network.send(param.grad)
                param.grad = megatron_integration.lossy_network.receive(param.grad, mask)
    
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def evaluate_with_megatron(model, eval_loader, device, is_megatron_model):
    """Evaluation function that works with both Megatron and existing models."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if is_megatron_model:
                # Megatron model evaluation
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                outputs = model(input_ids, attention_mask)
                logits = model.classifier(outputs)
            else:
                # Your existing HuggingFace model
                logits = model(**batch).logits
            
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return accuracy_score(all_labels, all_preds)


def train_to_accuracy_with_megatron(args):
    """
    Your existing training loop enhanced with Megatron-LM integration.
    This preserves all your experimental setup while using optimized tensor parallelism.
    """
    
    # Your existing distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Try NCCL first, fall back to gloo if needed
    try:
        dist.init_process_group(backend='nccl', init_method='env://')
    except:
        dist.init_process_group(backend='gloo', init_method='env://')
    
    print(f"[Rank {local_rank}] world_size = {dist.get_world_size()}")
    world_size = args.tensor_parallel_size
    
    # Validate world size
    actual_world_size = dist.get_world_size()
    if actual_world_size != world_size:
        print(f"Warning: actual world_size ({actual_world_size}) != tensor_parallel_size ({world_size})")
        world_size = actual_world_size
    
    group = dist.group.WORLD
    
    # Your existing configuration loading
    with open("src/config.yaml") as cf:
        dataset_config = yaml.safe_load(cf)[args.dataset]
    
    num_labels = dataset_config['num_labels']
    args.target_accuracy = dataset_config['target_acc']
    report_ttac = dataset_config.get('report_ttac', [])
    
    # Your existing output directory setup
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)
    open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
    log_file = os.path.join(args.output_dir, 'training.log')
    log_f = open(log_file, 'w')
    metrics = []
    
    # Your existing wandb setup
    if local_rank == 0:
        run_id = os.path.basename(os.path.abspath(args.output_dir))
        wandb.init(
            project="lossy_ai_networks_megatron",  # Updated project name
            name=run_id,
            config=vars(args),
            tags=["megatron", "tensor_parallel", "lossy_communication"]
        )
    
    # Initialize lossy network using your existing setup
    lossy_network = setup_lossy_network(args)
    
    # Create Megatron integration
    megatron_integration = MegatronLossyIntegration(args, lossy_network)
    
    # Try to enable Megatron integration
    megatron_available = megatron_integration.enable_lossy_communication()
    
    # Create model (Megatron or fallback to your existing approach)
    model, is_megatron_model = create_megatron_model(args, num_labels)
    
    if not is_megatron_model:
        # Fall back to your existing tensor parallelism setup
        print("📝 Using your existing custom tensor parallelism")
        
        # Your existing tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))
        
        # Your existing parallel layer replacement
        if hasattr(model, 'model'):
            backbone = model.model
        elif hasattr(model, 'transformer'):
            backbone = model.transformer
        else:
            backbone = model
        
        # Import and use your existing parallel layers
        from parallel_layers_gpt import RowParallelLinear
        from transformers.models.gpt2.modeling_gpt2 import Conv1D
        
        def replace_linears(module, world_size, group):
            """Your existing replace_linears function"""
            for name, child in list(module.named_children()):
                if isinstance(child, (nn.Linear, Conv1D)):
                    W = child.weight.data
                    if isinstance(child, Conv1D):
                        in_f, out_f = W.shape
                    else:
                        out_f, in_f = W.shape
                    
                    if out_f % world_size == 0:
                        wrapped = RowParallelLinear(child, world_size, group)
                        setattr(module, name, wrapped)
                        print(f"Parallelized layer {name}: [{out_f}, {in_f}] -> RowParallel")
                    else:
                        print(f"Skipped layer {name}: output dim {out_f} not divisible by world_size {world_size}")
                else:
                    replace_linears(child, world_size, group)
        
        replace_linears(backbone, world_size, group)
        print(backbone)
    else:
        # For Megatron model, tokenizer is handled differently
        tokenizer = AutoTokenizer.from_pretrained('gpt2')  # Use GPT-2 tokenizer for Megatron
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    model.to(torch.cuda.current_device())
    
    # Your existing dataset setup
    train_ds, eval_ds = get_dataset(args, tokenizer)
    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    eval_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])
    
    # Your existing data loaders
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size,
                                      rank=local_rank, shuffle=False)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, sampler=eval_sampler)
    
    # Your existing training setup
    use_fp16 = getattr(args, "fp16", False)
    scaler = GradScaler() if use_fp16 else None
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=getattr(args, 'weight_decay', 0.0))
    
    # Your existing training loop variables
    best_acc, no_imp, step = 0.0, 0, 0
    last_accuracies = []
    start = time.time()
    
    print(f"🚀 Starting training with {'Megatron-LM' if is_megatron_model else 'Custom'} tensor parallelism")
    if megatron_available:
        print("✅ Lossy communication integrated at Megatron tensor parallel level")
    else:
        print("⚠️ Using gradient-level lossy communication (fallback)")
    
    # Your existing training loop
    while step < args.max_steps:
        train_sampler.set_epoch(step)
        for batch in tqdm(train_loader, desc=f"Rank {local_rank} Step {step}",
                          disable=(local_rank != 0)):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            if use_fp16:
                # FP16 training path
                model.train()
                with autocast():
                    if is_megatron_model:
                        input_ids = batch['input_ids']
                        attention_mask = batch.get('attention_mask', None)
                        outputs = model(input_ids, attention_mask)
                        logits = model.classifier(outputs)
                        labels = batch['labels']
                        loss_fn = nn.CrossEntropyLoss()
                        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        outputs = model(**batch)
                        loss = outputs.loss
                
                scaler.scale(loss).backward()
                
                # Apply gradient-level losses if Megatron integration is not available
                if not megatron_available:
                    for _, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            mask = lossy_network.send(param.grad)
                            param.grad = lossy_network.receive(param.grad, mask)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # Regular training path
                loss = train_step_with_megatron(model, batch, optimizer, megatron_integration, is_megatron_model)
            
            step += 1
            
            # Your existing evaluation logic
            if step % args.eval_steps == 0:
                stop_signal = torch.zeros(1, dtype=torch.uint8, device=model.device)
                
                if local_rank == 0:
                    acc = evaluate_with_megatron(model, eval_loader, model.device, is_megatron_model)
                    elapsed = time.time() - start
                    line = f"Step {step} | Acc {acc:.4f} | Time {elapsed:.1f}s"
                    print(line)
                    log_f.write(line + "\n")
                    log_f.flush()
                    
                    # Log metrics including lossy communication stats
                    lossy_stats = megatron_integration.get_stats()
                    metrics.append({
                        'step': step, 
                        'accuracy': acc, 
                        'time': elapsed,
                        'lossy_stats': lossy_stats
                    })
                    
                    wandb_log = {
                        'accuracy': acc, 
                        'step': step, 
                        'time': elapsed,
                        'integration_type': 'megatron' if megatron_available else 'custom',
                        **{f'lossy_{k}': v for k, v in lossy_stats.items()}
                    }
                    wandb.log(wandb_log)
                    
                    # Your existing accuracy tracking logic
                    last_accuracies.append(acc)
                    if len(last_accuracies) > 5:
                        last_accuracies.pop(0)
                    
                    avg_acc = sum(last_accuracies) / len(last_accuracies)
                    print(f"Last {len(last_accuracies)} accuracies: {last_accuracies} | Running Avg: {avg_acc:.4f}")
                    print(f"Lossy communication stats: {lossy_stats}")
                    
                    # Your existing stopping criteria
                    if len(last_accuracies) == 5 and avg_acc >= args.target_accuracy:
                        print(f"Running average of last 5 evaluations reached target accuracy {args.target_accuracy}. Stopping.")
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
                    megatron_integration.disable_lossy_communication()
                    return
    
    # Final cleanup
    if local_rank == 0:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as mf:
            json.dump(metrics, mf, indent=2)
        wandb.finish()
    log_f.close()
    megatron_integration.disable_lossy_communication()


if __name__ == "__main__":
    import argparse
    
    # Your existing argument parser with all the same options
    parser = argparse.ArgumentParser(description="Megatron-LM + Lossy Network Training")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=256)
    
    # Your existing Gilbert-Elliott loss parameters
    parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e'],
                        help='Type of packet loss simulation: "ber" for Bernoulli, "g-e" for Gilbert-Elliott')
    parser.add_argument('--ge_config', type=str, default='default',
                        help='Configuration ID for Gilbert-Elliott simulation (used in g_e_params.csv)')
    
    # Your existing training parameters
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='gpt2-large')  # Default to GPT-2 for compatibility
    parser.add_argument('--dataset', type=str, default='mnli')  # Your current dataset
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--loss_rate', type=float, default=0.001)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--target_accuracy', type=float, default=0.75)
    parser.add_argument('--eval_steps', type=int, default=20)  # Your current setting
    parser.add_argument('--patience', type=int, default=15)    # Your current setting
    parser.add_argument('--max_steps', type=int, default=500)  # Your current setting
    parser.add_argument('--output_dir', type=str, default='./output')
    
    args = parser.parse_args()
    
    # Your existing setup
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    open(os.path.join(args.output_dir, 'ttac_report.txt'), 'a').close()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run training with Megatron integration
    train_to_accuracy_with_megatron(args)
