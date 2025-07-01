#!/bin/bash

# Quick test script with minimal dataset for fast debugging

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=ens5
export NCCL_IB_DISABLE=1
export NCCL_NET_OFI_DISABLE=1
export NCCL_P2P_LEVEL=SYS

# Test with minimal configuration
echo "=== Quick Test Run ==="

MASTER_ADDR=localhost MASTER_PORT=12355 torchrun \
    --nproc_per_node 2 \
    --master_addr localhost \
    --master_port 12355 \
    src/pytorch_train_tp_gpt_clean.py \
      --tensor_parallel_size 2 \
      --loss_type            ber \
      --model_name           "gpt2" \
      --dataset              "sst2" \
      --batch_size           4 \
      --max_length           64 \
      --learning_rate        3e-5 \
      --weight_decay         0.01 \
      --loss_rate            0.001 \
      --seed                 1234 \
      --max_samples          100 \
      --target_accuracy      0.5 \
      --eval_steps           5 \
      --patience             3 \
      --max_steps            20 \
      --output_dir           "test_output"

echo "=== Quick test completed ==="
