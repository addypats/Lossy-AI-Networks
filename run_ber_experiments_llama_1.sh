#!/bin/bash
# MODEL="meta-llama/Llama-3.2-1B"
# MODEL="Qwen/Qwen2-1.5B"
MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# MODEL="google/gemma-1.1-2b-it"
# MODEL="gpt2-large"
# MODEL="EleutherAI/gpt-neo-1.3B"
# DATASET="mnli"
# DATASET="hotpotqa"
# DATASET="squad"
DATASET="piqa"
# DATASET="arc"
# DATASET="sst2"
# DATASET="quality"

# LOSS_RATES=("0.0" "0.001" "0.002" "0.005" "0.01")
# LOSS_RATES=("0.001" "0.002" "0.005" "0.01")
# LOSS_RATES=("0.01")
# LOSS_RATES=()
# NUM_NODES=("2" "4" "8" "10")
NUM_NODES=("8" "10")
# NUM_NODES=("10")
# SEEDS=("10" "20" "30" "40" "50")
SEEDS=(10)

# CONFIGS=("one_precent" "half_percent" "point2_percent" "long_point1_percent" "short_1percent" "short_half_percent" "short_point_2percent" "short_point1_percent")
# CONFIGS=("half_percent" "point2_percent" "long_point1_percent" "short_1percent" "short_half_percent" "short_point_2percent" "short_point1_percent")
# CONFIGS=("one_precent")
CONFIGS=("short_half_percent")
# CONFIGS=("half_percent" "long_point1_percent" "short_1percent" "short_half_percent" "short_point1_percent")

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="lossy_net_burst_study"

# Create output directory if it doesn't exist
mkdir -p output_piqa

echo "Starting ber experiments!"

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="${nodes}nodes_${DATASET}_lr${loss_rate}_seed${seed}"
      output_dir="output_piqa/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        --batch_size $((8 * ${nodes})) \
        --learning_rate 1e-5 \
        --eval_steps 50 \
        --run_id "$run_id" \
        --epochs 20 \
        --seed "$seed" \
        --output_dir "$output_dir"
      
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All ber experiments completed!"

echo "Starting ge experiments!"

for config in "${CONFIGS[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="ge_${MODEL_ALIAS}_${nodes}nodes_${DATASET}_lr_${config}_seed${seed}"
      output_dir="output_piqa/ge_${MODEL_ALIAS}_output/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --num_nodes "$nodes" \
        --batch_size $((16 * ${nodes})) \
        --learning_rate 5e-6 \
        --run_id "$run_id" \
        --epochs 4 \
        --seed "$seed" \
        --output_dir "$output_dir" \
              --eval_steps 20 \
        --loss_type "g-e" \
        --ge_config "$config"
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All ge experiments completed!"

echo "All experiments completed!"
