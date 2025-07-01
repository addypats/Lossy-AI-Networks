#!/bin/bash

# Pure Tensor Parallelism experiments with lossy networks
# Uses the native TP implementation (no external dependencies)

export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=ens5
export NCCL_IB_DISABLE=1
export NCCL_NET_OFI_DISABLE=1
export NCCL_P2P_LEVEL=SYS

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Torchrun binary
TORCHRUN=$(which torchrun)

# Rendezvous settings
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12357

# GPUs to use
export CUDA_VISIBLE_DEVICES=0,1

# Tensor-parallel world size
TP_SIZE=(2)

# GilbertElliot Loss Model params
GE_CONFIG=("zero")

# Datasets
DATASETS=("mnli")

# Precision Flags
FP_FLAGS=(fp32)

# Iterations
ITERATIONS=(1)

# Ensure output directory exists
mkdir -p output_pure_tp_lossy_mnli

echo "=== Running Pure Tensor Parallelism Experiments ==="

for iter in "${ITERATIONS[@]}"; do
  echo
  echo "=== Starting iteration ${iter} ==="
  echo
  for temp_flag in "${FP_FLAGS[@]}"; do
  echo
    # Decide on the actual flag to pass into the Python script
    if [ "$temp_flag" = "fp16" ]; then
      fp_flag="--fp16"
      echo
      echo "=== Starting with precision ${fp_flag} ===\n"
      echo
    else
      fp_flag=""   # no flag for fp32
      echo
      echo "=== Starting with precision --fp32 ===\n"
      echo
    fi
    for tp_size in "${TP_SIZE[@]}"; do
    echo
      echo "=== Starting pure tensor parallelism with size ${tp_size} ==="
      echo
      for dataset in "${DATASETS[@]}"; do
      echo
        echo "=== Starting with dataset ${dataset} ==="
        echo
        for ge_config in "${GE_CONFIG[@]}"; do
          run_id="pure_tp_gpt2-large_precision-${temp_flag}_TP_Size-${tp_size}_ge_config_${ge_config}_Pure_TP_Iteration_${iter}"
          echo
          echo "=== Starting $run_id ==="
          echo

          # Pure TP Script
          echo "Using tensor_parallel_size: $tp_size"
          echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
          echo "Model weights will be SPLIT across GPUs"
          
          $TORCHRUN \
            --nproc_per_node $tp_size \
            --master_addr   $MASTER_ADDR \
            --master_port   $MASTER_PORT \
            src/pytorch_train_pure_tp_native.py \
              --tensor_parallel_size $tp_size \
              --loss_type            g-e \
              --ge_config            $ge_config \
              --model_name           "gpt2-large" \
              --dataset              $dataset \
              --batch_size           8 \
              --max_length           128 \
              --learning_rate        3e-5 \
              --weight_decay         0.01 \
              --loss_rate            0.001 \
              $fp_flag \
              --seed                 1234 \
              --max_samples          2000 \
              --target_accuracy      0.75 \
              --eval_steps           20 \
              --patience             15 \
              --max_steps            500 \
              --output_dir           "output_pure_tp_lossy_mnli/$run_id" \

          echo "=== Completed $run_id ==="
          echo
          echo
        done
      done
    done
  done
done

echo "All pure tensor-parallel runs completed."
