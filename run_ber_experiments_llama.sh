#!/bin/bash
MODEL="meta-llama/Llama-3.2-1B"
# MODEL="Qwen/Qwen2-1.5B"
# MODEL="microsoft/phi-2"
# MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# MODEL="google/gemma-1.1-2b-it"
# DATASET="mnli"
# DATASET="hotpotqa"
DATASET="squad"
# DATASET="piqa"

LOSS_RATES=("0.0" "0.001" "0.002" "0.005" "0.01")
# LOSS_RATES=(0.0)
NUM_NODES=("2" "4" "8" "10")
# NUM_NODES=(2)
# SEEDS=("10" "20" "30" "40" "50")
SEEDS=(10)

# GPU settings
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="lossy_network_latest"

# Create output directory if it doesn't exist
mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="${nodes}nodes_${DATASET}_lr${loss_rate}_seed${seed}"
      output_dir="output/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        --batch_size $((16 * ${nodes})) \
        --learning_rate 2e-5 \
        --eval_steps 20\
        --run_id "$run_id" \
        --epochs 7 \
        --seed "$seed" \
        --output_dir "$output_dir"
      
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All experiments completed!" 
