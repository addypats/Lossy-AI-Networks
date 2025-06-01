#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
GPUS_PER_NODE=2


torchrun \
  --nnodes=1 \
  --nproc-per-node=$GPUS_PER_NODE \
  --rdzv_id=test_rpc2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29500 \
  python test_rpc.py

