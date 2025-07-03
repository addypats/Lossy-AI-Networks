#!/usr/bin/env python3
"""
Megatron GPT2-Large Fine-tuning Script for MNLI Dataset
Uses original Megatron framework with tensor parallelism for fine-tuning GPT2-large on MNLI.
Logs to Weights & Biases and uses running average of last 5 evaluations for convergence.
"""

import os
import sys
import time
import json
import yaml
import argparse
import torch
import wandb
import numpy as np
from collections import deque

# Add Megatron to path
megatron_path = os.path.join(os.path.dirname(__file__), "Megatron-LM")
sys.path.append(megatron_path)

try:
    from megatron import get_args, get_timers, get_tokenizer, print_rank_0
    from megatron.core import mpu
    from megatron.data.gpt_dataset import build_train_valid_test_datasets
    from megatron.model import GPTModel
    from megatron.training import pretrain
    from megatron.utils import get_ltor_masks_and_position_ids
    from megatron.arguments import core_transformer_config_from_args
    from megatron.global_vars import set_global_variables
    from megatron.initialize import initialize_megatron
    from megatron.checkpointing import load_checkpoint, save_checkpoint
    from megatron.text_generation import generate_and_post_process
except ImportError as e:
    print(f"Error importing Megatron modules: {e}")
    print("Please ensure Megatron-LM is properly installed and accessible.")
    sys.exit(1)

from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer


class MNLIFinetuner:
    """Class to handle MNLI fine-tuning with Megatron GPT2-large"""
    
    def __init__(self, args):
        self.args = args
        self.accuracy_history = deque(maxlen=5)  # Store last 5 accuracies
        self.best_accuracy = 0.0
        self.eval_step = 0
        self.start_time = time.time()
        
        # Initialize Weights & Biases
        if torch.distributed.get_rank() == 0:
            wandb.init(
                project="megatron-gpt2-mnli-finetune",
                name=f"gpt2-large-mnli-tp{args.tensor_model_parallel_size}",
                config=vars(args)
            )
    
    def prepare_mnli_data(self):
        """Load and prepare MNLI dataset"""
        print_rank_0("Loading MNLI dataset...")
        
        # Load MNLI dataset
        dataset = load_dataset("glue", "mnli")
        train_dataset = dataset["train"]
        valid_matched = dataset["validation_matched"]
        valid_mismatched = dataset["validation_mismatched"]
        
        # Use validation_matched for evaluation
        return train_dataset, valid_matched
    
    def preprocess_mnli_batch(self, batch, tokenizer):
        """Preprocess MNLI batch for GPT2 input format"""
        premises = batch["premise"]
        hypotheses = batch["hypothesis"]
        labels = batch["label"]
        
        # Create input text in format: "premise [SEP] hypothesis"
        texts = []
        for premise, hypothesis in zip(premises, hypotheses):
            text = f"{premise} [SEP] {hypothesis}"
            texts.append(text)
        
        # Tokenize
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.args.seq_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }
    
    def evaluate_model(self, model, eval_dataset, tokenizer):
        """Evaluate model on MNLI validation set"""
        model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        print_rank_0(f"Starting evaluation at step {self.eval_step}...")
        
        with torch.no_grad():
            # Process evaluation data in batches
            batch_size = self.args.micro_batch_size
            for i in range(0, len(eval_dataset), batch_size):
                batch_end = min(i + batch_size, len(eval_dataset))
                batch_data = eval_dataset[i:batch_end]
                
                # Preprocess batch
                batch = self.preprocess_mnli_batch(batch_data, tokenizer)
                
                # Move to GPU
                input_ids = batch["input_ids"].cuda()
                attention_mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()
                
                # Forward pass
                logits = model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss (cross-entropy for classification)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, 3), labels.view(-1))  # 3 classes for MNLI
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Break after reasonable number of samples for faster evaluation
                if num_batches >= 100:  # Evaluate on ~100 batches
                    break
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return accuracy, avg_loss
    
    def log_metrics(self, accuracy, loss, learning_rate, step):
        """Log metrics to console and Weights & Biases"""
        elapsed_time = time.time() - self.start_time
        
        # Add to accuracy history
        self.accuracy_history.append(accuracy)
        running_avg = sum(self.accuracy_history) / len(self.accuracy_history)
        
        # Update best accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        # Log to console
        print_rank_0(f"Step {step:6d} | "
                    f"Accuracy: {accuracy:.4f} | "
                    f"Running Avg: {running_avg:.4f} | "
                    f"Loss: {loss:.6f} | "
                    f"LR: {learning_rate:.2e} | "
                    f"Time: {elapsed_time:.1f}s")
        
        # Log to Weights & Biases
        if torch.distributed.get_rank() == 0:
            wandb.log({
                "accuracy": accuracy,
                "running_accuracy": running_avg,
                "best_accuracy": self.best_accuracy,
                "eval_loss": loss,
                "learning_rate": learning_rate,
                "step": step,
                "elapsed_time": elapsed_time
            })
        
        return running_avg
    
    def check_convergence(self):
        """Check if running average has reached target accuracy"""
        if len(self.accuracy_history) == 5:  # Have 5 evaluations
            running_avg = sum(self.accuracy_history) / len(self.accuracy_history)
            if running_avg >= self.args.target_accuracy:
                print_rank_0(f"Convergence achieved! Running average {running_avg:.4f} >= target {self.args.target_accuracy:.4f}")
                return True
        return False


def model_provider(pre_process=True, post_process=True):
    """Build the model for fine-tuning"""
    args = get_args()
    
    print_rank_0("Building GPT2-large model for fine-tuning...")
    
    config = core_transformer_config_from_args(args)
    
    # Create model with classification head for MNLI (3 classes)
    model = GPTModel(
        config=config,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )
    
    # Add classification head
    if post_process:
        model.classification_head = torch.nn.Linear(config.hidden_size, 3)  # 3 classes for MNLI
    
    return model


def forward_step(data_iterator, model):
    """Forward step for training"""
    args = get_args()
    timers = get_timers()
    
    # Get the batch
    timers('batch-generator', log_level=2).start()
    try:
        data = next(data_iterator)
    except StopIteration:
        return None
    timers('batch-generator').stop()
    
    # Extract batch data
    input_ids = data['input_ids'].cuda()
    attention_mask = data['attention_mask'].cuda()
    labels = data['labels'].cuda()
    
    # Forward pass
    logits = model(input_ids, attention_mask=attention_mask)
    
    # Calculate loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, 3), labels.view(-1))
    
    return loss


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Provide train, validation, and test datasets"""
    args = get_args()
    
    # Load MNLI dataset
    dataset = load_dataset("glue", "mnli")
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation_matched"]
    
    # Convert to format expected by Megatron
    # This is a simplified version - you may need to adapt based on your Megatron version
    return train_dataset, valid_dataset, None


def main():
    """Main training loop"""
    parser = argparse.ArgumentParser(description="Megatron GPT2-Large MNLI Fine-tuning")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="gpt2-large",
                       help="Model name or path")
    parser.add_argument("--num-layers", type=int, default=36,
                       help="Number of transformer layers")
    parser.add_argument("--hidden-size", type=int, default=1280,
                       help="Hidden size of transformer")
    parser.add_argument("--num-attention-heads", type=int, default=20,
                       help="Number of attention heads")
    parser.add_argument("--seq-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--max-position-embeddings", type=int, default=1024,
                       help="Maximum position embeddings")
    
    # Training arguments
    parser.add_argument("--micro-batch-size", type=int, default=4,
                       help="Micro batch size per GPU")
    parser.add_argument("--global-batch-size", type=int, default=32,
                       help="Global batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--lr-decay-style", type=str, default="cosine",
                       help="Learning rate decay style")
    parser.add_argument("--lr-warmup-fraction", type=float, default=0.1,
                       help="Learning rate warmup fraction")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--clip-grad", type=float, default=1.0,
                       help="Gradient clipping")
    
    # Parallelism arguments
    parser.add_argument("--tensor-model-parallel-size", type=int, default=2,
                       help="Tensor model parallel size")
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1,
                       help="Pipeline model parallel size")
    
    # Training control
    parser.add_argument("--train-iters", type=int, default=10000,
                       help="Maximum training iterations")
    parser.add_argument("--eval-interval", type=int, default=20,
                       help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=500,
                       help="Save interval")
    parser.add_argument("--target-accuracy", type=float, default=0.75,
                       help="Target accuracy for convergence")
    
    # I/O arguments
    parser.add_argument("--save", type=str, default="./checkpoints",
                       help="Save directory")
    parser.add_argument("--load", type=str, default=None,
                       help="Load checkpoint directory")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Data path")
    parser.add_argument("--vocab-file", type=str, default=None,
                       help="Vocabulary file")
    parser.add_argument("--merge-file", type=str, default=None,
                       help="Merge file for BPE")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=1234,
                       help="Random seed")
    parser.add_argument("--fp16", action="store_true",
                       help="Use fp16 mixed precision")
    parser.add_argument("--use-checkpoint-opt_param-scheduler", action="store_true",
                       help="Use checkpoint optimizer and scheduler")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Log interval")
    
    args = parser.parse_args()
    
    # Initialize Megatron
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
    
    # Create fine-tuner instance
    finetuner = MNLIFinetuner(args)
    
    # Start pre-training with custom evaluation
    def evaluation_step():
        """Custom evaluation step"""
        model = get_args().model
        tokenizer = get_tokenizer()
        
        # Load evaluation dataset
        _, eval_dataset = finetuner.prepare_mnli_data()
        
        # Evaluate
        accuracy, loss = finetuner.evaluate_model(model, eval_dataset, tokenizer)
        
        # Log metrics
        learning_rate = get_args().optimizer.param_groups[0]['lr']
        running_avg = finetuner.log_metrics(accuracy, loss, learning_rate, finetuner.eval_step)
        
        finetuner.eval_step += 1
        
        # Check convergence
        if finetuner.check_convergence():
            print_rank_0("Target accuracy achieved. Stopping training.")
            return True
        
        return False
    
    # Run pre-training (fine-tuning) with custom hooks
    pretrain(
        train_valid_test_datasets_provider=train_valid_test_datasets_provider,
        model_provider=model_provider,
        ModelType=None,
        forward_step_func=forward_step,
        extra_args_provider=None
    )
    
    # Finish Weights & Biases logging
    if torch.distributed.get_rank() == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
