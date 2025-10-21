#!/bin/bash
# Launch 4 ranks for tensor parallelism (one per GPU)

# This is for the 4 GPU instnace machine
# source /home/ubuntu/tp-env/bin/activate

# This is for the 4 GPU instnace machine
# source /home/ubuntu/tp-venv/bin/activate

# # NCCL tuning
# export NCCL_DEBUG=WARN
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# # This is for the 4 GPU instance. Use this with the g5.12xlarge.
# # !–– New NCCL fixes ––!
# export NCCL_NET_OFI_DISABLE=1
# export NCCL_SOCKET_IFNAME=ens5
# export NCCL_IB_DISABLE=1
# export NCCL_LAUNCH_TIMEOUT=1200
# export NCCL_TIMEOUT=1200

# This is for the 8 GPU instance. Use this with the g5.48xlarge.
# Rendezvous

# Force NCCL over TCP, disable IB/OFI if you don’t have EFA
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=ens5      # ← your chosen interface
export NCCL_IB_DISABLE=1            # disable Infiniband
export NCCL_NET_OFI_DISABLE=1       # disable OFI
export NCCL_P2P_LEVEL=SYS           # prefer system-level P2P

# (Optional) for debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Just wandb things
# gxport WANDB_MODE=disabled


# export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Torchrun binary
TORCHRUN=$(which torchrun)

# Rendezvous settings
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12356

# GPUs to use
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Tensor-parallel world size
# TP_SIZE=(2 4 8)
TP_SIZE=(4)

# GilbertElliot Loss Model params
# GE_CONFIG=("default" "one" "one_precent" "half_percent" "point2_percent")
GE_CONFIG=("long_1percent" "long_half_percent" "long_point2_percent" "long_point1_percent")
# GE_CONFIG=("one_precent" "half_percent" "point2_percent")
# GE_CONFIG=("default" "one_precent" "half_percent")
# GE_CONFIG=("zero" "short_1percent" "short_half_percent" "short_point_2percent" "short_point1_percent")
# GE_CONFIG=("long_point1_percent")
# GE_CONFIG=("zero" "ber_20" "long_50percent" "90_loss")

# Loss-rate grid
LOSS_RATES=(0.001 0.002 0.005 0.01)
# LOSS_RATES=(0)

# Datasets
# DATASETS=("winogrande" "mnli" "hellaswag" "piqa")
DATASETS=("mnli")

# Precision Flags
# FP_FLAGS=(fp32 fp16)
FP_FLAGS=(fp32)

# To run the number of iterations
# ITERATIONS=(1 2 3 4 5)
ITERATIONS=(1)

# Ensure output directory exists
# mkdir -p output_Llama3.2-1B
mkdir -p output_gpt2-large_BurstyLosses_mnli


# Running script for uniform loss with loss rates like the previous ones - used for bernoulli' (The standard loss rate function)

# for iter in "${ITERATIONS[@]}"; do
#   echo
#   echo "=== Starting iteration ${iter} ==="
#   echo
#   for temp_flag in "${FP_FLAGS[@]}"; do
#   echo
#     # Decide on the actual flag to pass into the Python script
#     if [ "$temp_flag" = "fp16" ]; then
#       fp_flag="--fp16"
#       echo
#       echo "=== Starting with precision ${fp_flag} ===\n"
#       echo
#     else
#       fp_flag=""   # no flag for fp32
#       echo
#       echo "=== Starting with precision --fp32 ===\n"
#       echo
#     fi
#     for tp_size in "${TP_SIZE[@]}"; do
#     echo
#       echo "=== Starting tensor parallelism with size ${tp_size} ==="
#       echo
#       for dataset in "${DATASETS[@]}"; do
#       echo
#         echo "=== Starting with dataset ${dataset} ==="
#         echo
#         for loss_rate in "${LOSS_RATES[@]}"; do
#           run_id="target_steps_tp_gpt2-large_precision-${temp_flag}_Num_Nodes-${tp_size}_lr${loss_rate}_Iteration_${iter}"
#           echo
#           echo "=== Starting $run_id ==="
#           echo

#           # Original Script
#           $TORCHRUN \
#             --nproc_per_node $tp_size \
#             --master_addr   $MASTER_ADDR \
#             --master_port   $MASTER_PORT \
#             src/pytorch_train_tp_gpt.py \
#               --tensor_parallel_size $tp_size \
#               --loss_type             ber \
#               --ge_config             default \
#               --model_name           "gpt2-large" \
#               --dataset              $dataset \
#               --batch_size           16 \
#               --max_length           128 \
#               --learning_rate        3e-5 \
#               --weight_decay         0.01 \
#               --loss_rate            $loss_rate \
#               $fp_flag \
#               --seed                 1234 \
#               --max_samples          0 \
#               --target_accuracy      0.75 \
#               --eval_steps           20 \
#               --patience             10 \
#               --max_steps            500 \
#               --output_dir           "output_gpt2-large_uniform_Bernoulli_Losses_sst2/$run_id" \

#           echo "=== Completed $run_id ==="
#           echo
#           echo
#         done
#       done
#     done
#   done
# done

# Running script for Bursty loss with GilbertElliot Config

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
      echo "=== Starting tensor parallelism with size ${tp_size} ==="
      echo
      for dataset in "${DATASETS[@]}"; do
      echo
        echo "=== Starting with dataset ${dataset} ==="
        echo
        for ge_config in "${GE_CONFIG[@]}"; do
          run_id="target_steps_tp_gpt2-large_precision-${temp_flag}_Num_Nodes-${tp_size}_ge_config_${ge_config}_Long_Burst_Loss_Iteration_${iter}"
          echo
          echo "=== Starting $run_id ==="
          echo

          # Original Script
          $TORCHRUN \
            --nproc_per_node $tp_size \
            --master_addr   $MASTER_ADDR \
            --master_port   $MASTER_PORT \
            src/pytorch_train_tp_gpt.py \
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
              --max_samples          0 \
              --target_accuracy      0.75 \
              --eval_steps           20 \
              --patience             15 \
              --max_steps            500 \
	      --wandb
              --output_dir           "output_gpt2-large_BurstyLosses_mnli/$run_id" \

          echo "=== Completed $run_id ==="
          echo
          echo
        done
      done
    done
  done
done

echo "All tensor-parallel runs done."


