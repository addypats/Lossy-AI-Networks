#!/bin/bash
#MODEL="meta-llama/Llama-3.2-1B"
MODEL="openai-community/gpt2-large"
MODEL_ALIAS="gpt2-large"
DATASET="mnli"

# LOSS_RATES=("0.0" "0.001" "0.005" "0.01")
# NUM_NODES=("8")
SEEDS=("10" "20" "30" "40" "50")
# CONFIGS=("short_1percent" "short_half_percent" "short_point_2percent")


# GPUs to use
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


DP_SIZE=(4)

# Torchrun binary
TORCHRUN=$(which torchrun)

# Rendezvous settings
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

# GilbertElliot Loss Model params
# CONFIG=("one_precent" "half_percent" "point2_percent" "long_point1_percent")
CONFIG=("short_1percent" "short_half_percent" "short_point_2percent" "short_point1_percent")
# CONFIG=("long_point1_percent")


export WANDB_PROJECT="new_lossy_network"

mkdir -p dp_output_gpt2-large_BurstyLosses_mnli


# Original DP script from Pegah

# for config in "${CONFIGS[@]}"; do
#   for nodes in "${DP_SIZE[@]}"; do
#     for seed in "${SEEDS[@]}"; do
#       run_id="dp_ge_${MODEL_ALIAS}_${nodes}nodes_${DATASET}_lr_${config}_seed${seed}"
#       output_dir="dp_ge_${MODEL_ALIAS}_output/${DATASET}"
#       echo "Starting experiment: $run_id"
#       # Run the experiment
#       python src/main.py \
#         --model_name "$MODEL" \
#         --dataset "$DATASET" \
#         --num_nodes "$nodes" \
#         --batch_size $((16 * ${nodes})) \
#         --learning_rate 2e-5 \
#         --run_id "$run_id" \
#         --epochs 4 \
#         --seed "$seed" \
#         --output_dir "$output_dir" \
# 	      --eval_steps 20 \
#         --loss_type "g-e" \
#         --ge_config "$config" 
#       echo "Completed experiment: $run_id"
#       echo "--------------------------------"
#     done
#   done
# done


# My DP script for AWS

for config in "${CONFIGS[@]}"; do
  for nodes in "${DP_SIZE[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="dp_ge_${MODEL_ALIAS}_${nodes}nodes_${DATASET}_lr_${config}_seed${seed}"
      output_dir="dp_ge_${MODEL_ALIAS}_output/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      $TORCHRUN \
        --nproc_per_node $tp_size \
        --master_addr   $MASTER_ADDR \
        --master_port   $MASTER_PORT \
         src/main.py \
            --model_name "$MODEL" \
            --dataset "$DATASET" \
            --num_nodes "$nodes" \
            --batch_size $((16 * ${nodes})) \
            --learning_rate 2e-5 \
            --run_id "$run_id" \
            --epochs 4 \
            --seed "$seed" \
            --output_dir "dp_output_gpt2-large_BurstyLosses_mnli/$output_dir/$run_id" \
            --eval_steps 20 \
            --loss_type "g-e" \
            --ge_config "$config" 
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All experiments completed!" 