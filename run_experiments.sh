#!/bin/bash

# Set the model name
MODEL="meta-llama/Llama-3.2-1B"
DATASET="winogrande"

# Arrays for the different parameters
LOSS_RATES=("0")
# LOSS_RATES=("0" "0.001" "0.005" "0.01")
NUM_NODES=("2" "4" "6")
PRECISION=("32")
# PRECISION=("16" "32")

# GPU settings
export CUDA_VISIBLE_DEVICES=0  # Set which GPU to use (0, 1, etc. or multiple like "0,1")

# Create output directory if it doesn't exist
mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for prec in "${PRECISION[@]}"; do
      # Generate a unique run ID
      run_id="llama_3_2_1b_${DATASET}_lr${loss_rate}_nodes${nodes}_fp${prec}_rep${rep}"
      echo "Starting experiment: $run_id"
      
      # Set fp16 flag based on precision
      fp16_flag=""
      if [ "$prec" == "16" ]; then
        fp16_flag="--fp16"
      fi
      
      # Run the experiment
      python src/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        $fp16_flag \
        --run_id "$run_id"
      
      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All experiments completed!" 