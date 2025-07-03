#!/bin/bash

# Megatron GPT2-Large MNLI Fine-tuning Script
# This script runs Megatron GPT2-large fine-tuning on MNLI dataset with tensor parallelism

set -e

# Configuration
MODEL_NAME="gpt2-large"
DATASET="mnli"
TARGET_ACCURACY=0.75
EVAL_INTERVAL=20
TENSOR_PARALLEL_SIZE=2  # Adjust based on your GPU count
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32
LEARNING_RATE=1e-5
MAX_STEPS=10000

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
MEGATRON_PATH="${SCRIPT_DIR}/Megatron-LM"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints/gpt2-large-mnli"
DATA_PATH="${SCRIPT_DIR}/data"

# Create necessary directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$DATA_PATH"

# Set up environment
export PYTHONPATH="${MEGATRON_PATH}:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPU setup

# Weights & Biases setup (optional)
# export WANDB_PROJECT="megatron-gpt2-mnli-finetune"
# export WANDB_ENTITY="your-wandb-username"  # Replace with your W&B username

echo "Starting Megatron GPT2-large fine-tuning on MNLI..."
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET"
echo "  Target Accuracy: $TARGET_ACCURACY"
echo "  Evaluation Interval: $EVAL_INTERVAL steps"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Micro Batch Size: $MICRO_BATCH_SIZE"
echo "  Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Steps: $MAX_STEPS"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo ""

# Run the fine-tuning script
python -m torch.distributed.launch \
    --nproc_per_node=$TENSOR_PARALLEL_SIZE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    megatron_gpt2_finetune.py \
    --model-name "$MODEL_NAME" \
    --num-layers 36 \
    --hidden-size 1280 \
    --num-attention-heads 20 \
    --seq-length 512 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr $LEARNING_RATE \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --pipeline-model-parallel-size 1 \
    --train-iters $MAX_STEPS \
    --eval-interval $EVAL_INTERVAL \
    --save-interval 500 \
    --target-accuracy $TARGET_ACCURACY \
    --save "$CHECKPOINT_DIR" \
    --data-path "$DATA_PATH" \
    --seed 1234 \
    --fp16 \
    --use-checkpoint-opt_param-scheduler \
    --log-interval 10

echo "Fine-tuning completed!"
