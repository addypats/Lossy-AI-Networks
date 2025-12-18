#!/bin/bash
set -euo pipefail

# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="Qwen/Qwen2-1.5B"
MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_ALIAS="TinyLlama"
DATASET="piqa"
# LOSS_RATES=("0" "0.005" "0.01")
# LOSS_RATES=("0" "0.005" "0.01")
# LOSS_RATES=("0.01")

# Testing
# LOSS_RATES=("1")

# NUM_NODES=("2" "4" "8" "10")
# NUM_NODES=("8" "10")
#NUM_NODES=("2")
# SEEDS=("10" "20" "30" "40" "50")
SEEDS=("10" "20" "30")
# SEEDS=(10)


# GPUs on this machine (e.g., 4 GPUs)
# GPUS_LIST=(1 2 4)
GPUS_LIST=(1)
#SEEDS=(1 2 3)

# Per-GPU batch size (HF Trainer interprets this as per_device_* batch size)
PER_DEVICE_BS=8
LR=1e-5
#EPOCHS=1
#EVAL_STEPS=50

# CONFIGS=("one_precent" "half_percent" "short_1percent" "short_half_percent")
# CONFIGS=("short_1percent" "short_half_percent")
# CONFIGS=("half_percent" "short_1percent" "short_half_percent")


# CONFIGS_DET=("high_persistence_low_intensity_1" "high_persistence_low_intensity_2" "high_persistence_low_intensity_3" "high_persistence_low_intensity_4" "high_persistence_low_intensity_5" "high_persistence_low_intensity_6" "high_intensity_low_persistence_1" "high_intensity_low_persistence_2" "high_intensity_low_persistence_3" "high_intensity_low_persistence_4" "high_intensity_low_persistence_5" "high_intensity_low_persistence_6")
CONFIGS_DET=("high_persistence_low_intensity_1" "high_persistence_low_intensity_2" "high_persistence_low_intensity_3")

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="lossy_dist_fsdp_study"


# Logging Directory
export SANITY_CHECK_LOGS=/home/ubuntu/Lossy-AI-Networks/sanity_check_logs

# Using Ring All-Reduce
export NCCL_ALGO=Ring

# Args for dist training
export MASTER_ADDR=172.31.12.217     # Node 0 private IP
export MASTER_PORT=29500
export NNODES=8
# export NPROC_PER_NODE=4

export NCCL_SOCKET_IFNAME=enp39s0   # same as above
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Create output directory if it doesn't exist
mkdir -p output_piqa

echo "Starting fsdp experiments!"

echo "Ber fsdp exp started"

for loss_rate in "${LOSS_RATES[@]}"; do
  for gpus in "${GPUS_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ts=$(date +%Y%m%d-%H%M%S)

      run_id="${gpus}gpus_${DATASET}_seed${seed}_loss-rate${loss_rate}_${ts}"
      output_dir="output_piqa/${DATASET}"

      echo "Starting experiment: $run_id"

      # Make run_id visible to Python code (lossy_patch.py)
      export RUN_ID="${run_id}"

      TORCH_LOGS="distributed,dist_fsdp" TORCH_DISTRIBUTED_DEBUG=DETAIL \
      torchrun --nnodes=$NNODES \
  	--node_rank=4 \
  	--master_addr=$MASTER_ADDR \
  	--master_port=$MASTER_PORT \
	--nproc_per_node="${gpus}" \
	src/main_fsdp.py \
        --model_name "${MODEL}" \
        --dataset "${DATASET}" \
        --run_id "${run_id}" \
        --batch_size "${PER_DEVICE_BS}" \
        --learning_rate "${LR}" \
        --eval_steps 20 \
        --epochs 20 \
        --loss_rate "$loss_rate" \
	--loss-enable-all \
        --seed "${seed}" \
        --output_dir "${output_dir}" \
        --fp16 \
	--num_nodes "${NNODES}"

      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done  
  done
done

echo "All ber fsdp exp completed"

echo "GE fsdp exp started"

for config in "${CONFIGS[@]}"; do
  for gpus in "${GPUS_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ts=$(date +%Y%m%d-%H%M%S)

      run_id="${gpus}gpus_${DATASET}_seed${seed}_loss-rate_${config}_${ts}"
      output_dir="output_piqa/${DATASET}"

      echo "Starting experiment: $run_id"

      # Make run_id visible to Python code (lossy_patch.py)
      export RUN_ID="${run_id}"

      TORCH_LOGS="distributed,dist_fsdp" TORCH_DISTRIBUTED_DEBUG=DETAIL \
      torchrun --nnodes=$NNODES \
        --node_rank=4 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --nproc_per_node="${gpus}" \
	src/main_fsdp.py \
        --model_name "${MODEL}" \
        --dataset "${DATASET}" \
        --run_id "${run_id}" \
        --batch_size "${PER_DEVICE_BS}" \
        --learning_rate "${LR}" \
        --eval_steps 20 \
        --epochs 20 \
        --loss_type "g-e" \
        --ge_config "$config" \
        --loss-enable-all \
        --seed "${seed}" \
        --output_dir "${output_dir}" \
        --fp16 \
	--num_nodes "${NNODES}"

      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All ge fsdp exp completed"

echo "Starting fsdp det experiments!"

for config in "${CONFIGS_DET[@]}"; do
  for gpus in "${GPUS_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
      ts=$(date +%Y%m%d-%H%M%S)

      run_id="${gpus}gpus_${DATASET}_seed${seed}_loss-rate_${config}_${ts}"
      output_dir="output_piqa/${DATASET}"

      echo "Starting experiment: $run_id"

      # Make run_id visible to Python code (lossy_patch.py)
      export RUN_ID="${run_id}"

      TORCH_LOGS="distributed,dist_fsdp" TORCH_DISTRIBUTED_DEBUG=DETAIL \
	torchrun --nnodes=$NNODES \
        --node_rank=4 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --nproc_per_node="${gpus}" \
        src/main_fsdp.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --batch_size "${PER_DEVICE_BS}" \
        --learning_rate "${LR}" \
        --run_id "$run_id" \
        --epochs 20 \
        --seed "$seed" \
        --output_dir "$output_dir" \
              --eval_steps 20 \
        --loss-enable-all \
        --loss_type "det" \
        --det_config "$config" \
        --num_nodes "${NNODES}"
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All fsdp det experiments completed"

echo "All fsdp experiments completed!"

# echo "All experiments completed!"
