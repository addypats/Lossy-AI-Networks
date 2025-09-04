#!/bin/bash
# MODEL="meta-llama/Llama-3.2-1B"
MODEL="openai-community/gpt2-large"
# MODEL_ALIAS="gpt2-large"
# MODEL="Qwen/Qwen2-1.5B"
# MODEL="google/gemma-1.1-2b-it"
# MODEL_ALIAS="llama-3.2-1b"
# MODEL="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# DATASET="mnli"
# DATASET="hotpotqa"
DATASET="squad"
# DATASET="piqa"

# LOSS_RATES=("0.0" "0.001" "0.005" "0.01")
NUM_NODES=("2" "4" "8" "10")
# NUM_NODES=("4")
# SEEDS=("10" "20" "30" "40" "50")
SEEDS=("10")
CONFIGS=("one_precent" "half_percent" "point2_percent" "long_point1_percent" "short_1percent" "short_half_percent" "short_point_2percent" "short_point1_percent")
# CONFIGS=("short_1percent")


export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="lossy_network_latest"

for config in "${CONFIGS[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_id="ge_${MODEL_ALIAS}_${nodes}nodes_${DATASET}_lr_${config}_seed${seed}"
      output_dir="ge_${MODEL_ALIAS}_output/${DATASET}"
      echo "Starting experiment: $run_id"
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --num_nodes "$nodes" \
        --batch_size $((16 * ${nodes})) \
        --learning_rate 2e-5 \
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

echo "All experiments completed!" 
