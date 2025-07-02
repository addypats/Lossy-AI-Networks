#!/bin/bash
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
torchrun --nproc_per_node=$NGPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    megatron_lossy_training.py \
    --tensor-model-parallel-size $TENSOR_PARALLEL_SIZE \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --model-size $MODEL_SIZE \
    --fp16 \
    --loss-rate 0.05 \
    --train-iters 1000 \
    --lr 1e-5 \
    --min-lr 1e-6 \
    --weight-decay 0.01 \
    --clip-grad 1.0
