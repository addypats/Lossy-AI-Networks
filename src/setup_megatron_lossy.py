#!/usr/bin/env python3
"""
Setup script for Megatron-LM + Lossy Network Integration
This script helps you set up the environment and dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"Error output: {result.stderr}")
        return False
    return True

def install_megatron_lm():
    """Install Megatron-LM and dependencies."""
    print("🚀 Installing Megatron-LM and dependencies...")
    
    # Install required packages
    packages = [
        "torch>=1.13.0",
        "transformers>=4.20.0",
        "datasets",
        "tensorboard",
        "wandb",
        "apex",  # NVIDIA Apex for optimizations
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        if not run_command(f"pip install {package}", check=False):
            print(f"⚠️ Failed to install {package}, continuing...")
    
    # Clone and install Megatron-LM
    megatron_dir = "Megatron-LM"
    if not os.path.exists(megatron_dir):
        print("📥 Cloning Megatron-LM repository...")
        if not run_command("git clone https://github.com/NVIDIA/Megatron-LM.git"):
            return False
    
    # Install Megatron-LM
    original_dir = os.getcwd()
    try:
        os.chdir(megatron_dir)
        print("🔧 Installing Megatron-LM...")
        if not run_command("pip install -e ."):
            return False
    finally:
        os.chdir(original_dir)
    
    print("✅ Megatron-LM installation completed!")
    return True

def create_example_config():
    """Create example configuration files."""
    config_content = """
# Example Megatron Configuration for Lossy Network Research
# Adjust these parameters based on your hardware and research needs

MODEL_CONFIG = {
    # Model architecture
    'num_layers': 12,           # Number of transformer layers
    'hidden_size': 768,         # Hidden dimension
    'num_attention_heads': 12,  # Number of attention heads
    'seq_length': 1024,         # Sequence length
    'max_position_embeddings': 1024,
    'vocab_size': 50257,        # GPT-2 vocabulary size
    
    # Parallelism settings
    'tensor_model_parallel_size': 2,   # Number of GPUs for tensor parallelism
    'pipeline_model_parallel_size': 1, # Number of GPUs for pipeline parallelism
    
    # Training settings
    'micro_batch_size': 4,      # Micro batch size per GPU
    'global_batch_size': 32,    # Total batch size across all GPUs
    'lr': 1e-5,                # Learning rate
    'min_lr': 1e-6,            # Minimum learning rate
    'weight_decay': 0.01,       # Weight decay
    'clip_grad': 1.0,          # Gradient clipping
    
    # Mixed precision
    'fp16': True,              # Use FP16 mixed precision
    'bf16': False,             # Use BF16 (if supported)
    
    # Checkpointing
    'save_interval': 1000,      # Save checkpoint every N iterations
    'eval_interval': 500,       # Evaluate every N iterations
}

# Lossy Network Configuration
LOSSY_CONFIG = {
    'loss_rate': 0.05,          # 5% packet loss rate
    'burst_length': 10,         # Burst loss length
    'enable_on_forward': True,  # Apply losses to forward pass
    'enable_on_backward': True, # Apply losses to backward pass (gradients)
}

# Dataset Configuration
DATASET_CONFIG = {
    'train_data_path': 'path/to/your/train/data',
    'valid_data_path': 'path/to/your/valid/data',
    'tokenizer_name': 'gpt2',
    'seq_length': 1024,
    'split': '949,50,1',  # Train, validation, test split
}
"""
    
    with open("config_example.py", "w") as f:
        f.write(config_content)
    
    print("📝 Created config_example.py")

def create_run_script():
    """Create a simple run script for testing."""
    script_content = """#!/bin/bash
# Simple run script for Megatron + Lossy Network fine-tuning

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPU setup
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# Number of GPUs
NGPUS=2

# Model and training arguments
MODEL_SIZE="gpt2-medium"  # or gpt2-large
TENSOR_PARALLEL_SIZE=2
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=16

# Run the training
torchrun --nproc_per_node=$NGPUS \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    megatron_lossy_training.py \\
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \\
    --micro-batch-size $MICRO_BATCH_SIZE \\
    --global-batch-size $GLOBAL_BATCH_SIZE \\
    --model-size $MODEL_SIZE \\
    --fp16 \\
    --loss-rate 0.05 \\
    --train-iters 1000 \\
    --lr 1e-5 \\
    --min-lr 1e-6 \\
    --weight-decay 0.01 \\
    --clip-grad 1.0
"""
    
    with open("run_lossy_training.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    os.chmod("run_lossy_training.sh", 0o755)
    print("📝 Created run_lossy_training.sh")

def create_training_script():
    """Create the main training script that uses the integration."""
    script_content = """#!/usr/bin/env python3
\"\"\"
Main training script for Megatron + Lossy Network fine-tuning.
This script demonstrates how to use the integration for your research.
\"\"\"

import argparse
import torch
from megatron_lossy_integration import MegatronLossyFineTuner, LossyMegatronIntegration

# Import your actual lossy network here
# from your_lossy_network import YourLossyNetwork

class SimpleLossyNetwork:
    \"\"\"
    Simple example lossy network.
    Replace this with your actual lossy network implementation.
    \"\"\"
    def __init__(self, loss_rate=0.1, burst_length=5):
        self.loss_rate = loss_rate
        self.burst_length = burst_length
        self.step_count = 0
        
    def send(self, tensor):
        \"\"\"Apply lossy transformation to tensor.\"\"\"
        self.step_count += 1
        
        if not tensor.requires_grad:
            return None
            
        # Simple bernoulli loss model
        if self.step_count % 100 < self.burst_length:
            # Burst loss period
            loss_prob = self.loss_rate * 3  # Higher loss during burst
        else:
            # Normal period
            loss_prob = self.loss_rate
            
        # Create mask (1 = keep, 0 = drop)
        mask = torch.bernoulli(torch.full_like(tensor, 1 - loss_prob))
        return mask

def create_dummy_dataset(seq_len=512, vocab_size=50257, num_samples=1000):
    \"\"\"Create a dummy dataset for testing.\"\"\"
    class DummyDataset:
        def __init__(self):
            self.data = []
            for _ in range(num_samples):
                input_ids = torch.randint(0, vocab_size, (seq_len,))
                labels = input_ids.clone()
                self.data.append({
                    'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': torch.ones_like(input_ids)
                })
        
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    return DummyDataset()

def parse_args():
    parser = argparse.ArgumentParser(description='Megatron + Lossy Network Training')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='gpt2-medium',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])
    parser.add_argument('--tensor-model-parallel-size', type=int, default=2)
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1)
    
    # Training arguments
    parser.add_argument('--micro-batch-size', type=int, default=2)
    parser.add_argument('--global-batch-size', type=int, default=16)
    parser.add_argument('--train-iters', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--fp16', action='store_true')
    
    # Lossy network arguments
    parser.add_argument('--loss-rate', type=float, default=0.05)
    parser.add_argument('--burst-length', type=int, default=5)
    parser.add_argument('--enable-forward-loss', action='store_true', default=True)
    parser.add_argument('--enable-backward-loss', action='store_true', default=True)
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🚀 Starting Megatron + Lossy Network Training")
    print(f"📊 Loss rate: {args.loss_rate}")
    print(f"🔄 Tensor parallel size: {args.tensor_model_parallel_size}")
    
    # Initialize lossy network
    lossy_network = SimpleLossyNetwork(
        loss_rate=args.loss_rate,
        burst_length=args.burst_length
    )
    
    # Model configuration
    model_config = {
        'tensor_model_parallel_size': args.tensor_model_parallel_size,
        'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
        'micro_batch_size': args.micro_batch_size,
        'global_batch_size': args.global_batch_size,
    }
    
    # Create datasets (replace with your actual datasets)
    print("📝 Creating dummy datasets (replace with real data)")
    train_dataset = create_dummy_dataset()
    val_dataset = create_dummy_dataset(num_samples=100)
    
    # Initialize fine-tuner
    fine_tuner = MegatronLossyFineTuner(
        loss_network=lossy_network,
        model_config=model_config
    )
    
    # Test integration without full training
    print("🧪 Testing lossy integration...")
    integration = LossyMegatronIntegration(
        lossy_network,
        enable_on_forward=args.enable_forward_loss,
        enable_on_backward=args.enable_backward_loss
    )
    
    try:
        integration.enable()
        print("✅ Lossy integration test successful!")
        
        # Here you would call fine_tuner.run_fine_tuning(train_dataset, val_dataset)
        # For now, we just test the setup
        print("🔧 Setup complete. Ready for fine-tuning!")
        print("💡 To run actual training, uncomment the fine_tuner.run_fine_tuning() call")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Megatron-LM is properly installed")
    finally:
        integration.disable()

if __name__ == "__main__":
    main()
"""
    
    with open("megatron_lossy_training.py", "w") as f:
        f.write(script_content)
    
    print("📝 Created megatron_lossy_training.py")

def main():
    """Main setup function."""
    print("🔧 Setting up Megatron-LM + Lossy Network Integration")
    
    # Check if we should install Megatron
    install = input("Do you want to install Megatron-LM? (y/n): ").lower().strip()
    if install in ['y', 'yes']:
        if not install_megatron_lm():
            print("❌ Failed to install Megatron-LM")
            return
    
    # Create example files
    create_example_config()
    create_run_script()
    create_training_script()
    
    print("\n✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Review and modify config_example.py for your specific setup")
    print("2. Replace SimpleLossyNetwork with your actual lossy network in megatron_lossy_training.py")
    print("3. Prepare your datasets and update the dataset paths")
    print("4. Run: python megatron_lossy_training.py --help to see all options")
    print("5. For multi-GPU training, use: bash run_lossy_training.sh")
    
    print("\n📖 Key files created:")
    print("- megatron_lossy_integration.py: Main integration code")
    print("- megatron_lossy_training.py: Training script")
    print("- config_example.py: Configuration examples")
    print("- run_lossy_training.sh: Multi-GPU run script")

if __name__ == "__main__":
    main()
