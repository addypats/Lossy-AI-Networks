#!/bin/bash

echo "Setting up ready-made tensor parallelism solutions..."

# Update requirements with ready-made TP libraries
echo "Installing dependencies..."

# Option 1: FairScale (Meta's tensor parallelism)
echo "Installing FairScale..."
pip install fairscale

# Option 2: Accelerate (Hugging Face distributed training)
echo "Installing Accelerate..."
pip install accelerate

# Option 3: Already have PyTorch DDP (comes with PyTorch)

echo "All dependencies installed!"

echo ""
echo "Available solutions:"
echo "1. FairScale TP: src/pytorch_train_fairscale.py"
echo "2. Accelerate: src/pytorch_train_accelerate.py" 
echo "3. Simple DDP: src/pytorch_train_simple_ddp.py (recommended)"

echo ""
echo "Quick test with Simple DDP (no extra dependencies needed):"
echo "MASTER_ADDR=localhost MASTER_PORT=12355 torchrun --nproc_per_node 2 --master_addr localhost --master_port 12355 src/pytorch_train_simple_ddp.py --model_name gpt2 --dataset sst2 --max_samples 100 --max_steps 10 --eval_steps 5"
