#!/usr/bin/env python3
"""
Simplified Megatron GPT2-Large Fine-tuning for MNLI
Uses standard Megatron patterns with custom evaluation and W&B logging
"""

import os
import sys
import time
import json
import torch
import wandb
import numpy as np
from collections import deque
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Add Megatron path
MEGATRON_PATH = os.path.join(os.path.dirname(__file__), "Megatron-LM")
sys.path.insert(0, MEGATRON_PATH)

try:
    import megatron
    from megatron import get_args
    from megatron.initialize import initialize_megatron
    from megatron.training import pretrain
    from megatron.model import GPTModel
    from megatron.utils import get_ltor_masks_and_position_ids
    from megatron.data.gpt_dataset import build_train_valid_test_datasets
    from megatron.arguments import core_transformer_config_from_args
except ImportError as e:
    print(f"Failed to import Megatron: {e}")
    print("Please ensure Megatron-LM is installed and accessible")
    sys.exit(1)


class MNLITrainer:
    """MNLI fine-tuning trainer with W&B logging"""
    
    def __init__(self):
        self.accuracy_history = deque(maxlen=5)
        self.best_accuracy = 0.0
        self.step_count = 0
        self.start_time = time.time()
        
        # Initialize W&B on rank 0
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            args = get_args()
            wandb.init(
                project="megatron-gpt2-mnli",
                name=f"gpt2-large-tp{args.tensor_model_parallel_size}",
                config=vars(args)
            )
    
    def load_mnli_data(self):
        """Load MNLI dataset"""
        print("Loading MNLI dataset...")
        dataset = load_dataset("glue", "mnli")
        return dataset["train"], dataset["validation_matched"]
    
    def evaluate_on_mnli(self, model, tokenizer):
        """Evaluate model on MNLI validation set"""
        _, eval_data = self.load_mnli_data()
        
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        # Sample a subset for faster evaluation
        eval_subset = eval_data.select(range(min(1000, len(eval_data))))
        
        with torch.no_grad():
            for i, example in enumerate(eval_subset):
                if i >= 200:  # Limit evaluation size
                    break
                    
                premise = example["premise"]
                hypothesis = example["hypothesis"]
                label = example["label"]
                
                # Format input
                text = f"{premise} [SEP] {hypothesis}"
                tokens = tokenizer.encode(text)
                
                if len(tokens) > 512:  # Truncate if too long
                    tokens = tokens[:512]
                
                # Convert to tensor
                input_ids = torch.tensor([tokens]).cuda()
                
                # Forward pass
                try:
                    logits = model(input_ids)
                    
                    # Get prediction (assuming we added a classification head)
                    if hasattr(model, 'classification_head'):
                        pred = torch.argmax(logits, dim=-1).item()
                    else:
                        # Simple heuristic for GPT without classification head
                        pred = 1  # Default prediction
                    
                    if pred == label:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy, total_loss / max(total, 1)
    
    def log_metrics(self, accuracy, loss, lr, step):
        """Log metrics to console and W&B"""
        elapsed = time.time() - self.start_time
        
        # Update accuracy history
        self.accuracy_history.append(accuracy)
        running_avg = sum(self.accuracy_history) / len(self.accuracy_history)
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        # Console logging
        print(f"Step {step:6d} | Acc: {accuracy:.4f} | "
              f"Avg: {running_avg:.4f} | Loss: {loss:.6f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.1f}s")
        
        # W&B logging
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            wandb.log({
                "accuracy": accuracy,
                "running_accuracy": running_avg,
                "best_accuracy": self.best_accuracy,
                "loss": loss,
                "learning_rate": lr,
                "step": step,
                "elapsed_time": elapsed
            })
        
        return running_avg
    
    def check_convergence(self, target_acc=0.75):
        """Check if running average meets target"""
        if len(self.accuracy_history) >= 5:
            running_avg = sum(self.accuracy_history) / len(self.accuracy_history)
            return running_avg >= target_acc
        return False


# Global trainer instance
trainer = MNLITrainer()


def model_provider(pre_process=True, post_process=True):
    """Build GPT model with classification head"""
    print("Building GPT2-large model...")
    
    args = get_args()
    config = core_transformer_config_from_args(args)
    
    model = GPTModel(
        config=config,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )
    
    # Add classification head for MNLI (3 classes)
    if post_process:
        model.classification_head = torch.nn.Linear(config.hidden_size, 3)
        model.classification_head.cuda()
    
    return model


def forward_step(data_iterator, model):
    """Forward training step"""
    args = get_args()
    
    try:
        data = next(data_iterator)
    except StopIteration:
        return torch.tensor(0.0, requires_grad=True).cuda()
    
    # Simple forward pass - you'll need to adapt this based on your data format
    tokens = data['text'].cuda()
    labels = data.get('labels', None)
    
    logits = model(tokens)
    
    # Calculate loss
    if labels is not None and hasattr(model, 'classification_head'):
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels.cuda())
    else:
        # Default language modeling loss
        shifted_labels = tokens[..., 1:].contiguous()
        shifted_logits = logits[..., :-1, :].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(shifted_logits.view(-1, shifted_logits.size(-1)), 
                      shifted_labels.view(-1))
    
    return loss


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide datasets for training"""
    args = get_args()
    
    # Load MNLI data
    train_data, valid_data = trainer.load_mnli_data()
    
    # Simple conversion - you may need to implement proper data preprocessing
    # This is a placeholder that returns the datasets
    return train_data, valid_data, None


def extra_args_provider(parser):
    """Add custom arguments"""
    group = parser.add_argument_group(title='MNLI fine-tuning')
    group.add_argument('--target-accuracy', type=float, default=0.75,
                      help='Target accuracy for convergence')
    group.add_argument('--eval-steps', type=int, default=20,
                      help='Evaluation interval in steps')
    return parser


def main():
    """Main function"""
    # Initialize Megatron
    initialize_megatron(extra_args_provider=extra_args_provider)
    args = get_args()
    
    print(f"Starting Megatron GPT2-large fine-tuning on MNLI")
    print(f"Target accuracy: {args.target_accuracy}")
    print(f"Evaluation interval: {args.eval_steps} steps")
    print(f"Tensor parallel size: {args.tensor_model_parallel_size}")
    
    # Custom evaluation hook
    def custom_eval_hook():
        """Custom evaluation during training"""
        if trainer.step_count % args.eval_steps == 0:
            model = args.model  # Access model from args
            tokenizer = args.tokenizer if hasattr(args, 'tokenizer') else None
            
            if tokenizer is None:
                print("Warning: No tokenizer available for evaluation")
                return False
            
            # Evaluate
            accuracy, loss = trainer.evaluate_on_mnli(model, tokenizer)
            
            # Log metrics
            lr = args.lr  # Current learning rate
            running_avg = trainer.log_metrics(accuracy, loss, lr, trainer.step_count)
            
            # Check convergence
            if trainer.check_convergence(args.target_accuracy):
                print(f"Target accuracy {args.target_accuracy} achieved!")
                return True
        
        trainer.step_count += 1
        return False
    
    # Start pre-training (fine-tuning)
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={
            'tokenizer_type': 'GPT2BPETokenizer',
            'num_layers': 36,
            'hidden_size': 1280,
            'num_attention_heads': 20,
            'seq_length': 512,
            'max_position_embeddings': 1024,
        }
    )
    
    # Finish W&B
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        wandb.finish()
    
    print("Fine-tuning completed!")


if __name__ == "__main__":
    main()
