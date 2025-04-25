#!/bin/bash

# Set the model name and dataset
MODEL="meta-llama/Llama-3.2-1B"
DATASET="winogrande"

# Arrays for different parameters
LOSS_RATES=("0.001" "0.005")
NUM_NODES=("2")
PRECISION=("16")

# Set GPU (modify as needed)
export CUDA_VISIBLE_DEVICES=0

# Output directory setup
mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for prec in "${PRECISION[@]}"; do
      run_id="llama_tp_${DATASET}_lr${loss_rate}_nodes${nodes}_fp${prec}_$(date +%s)"
      echo "Starting experiment: $run_id"

      # Set FP16 flag
      fp16_flag=""
      if [ "$prec" == "16" ]; then
        fp16_flag="--fp16"
      fi

      # Launch with DeepSpeed
      deepspeed src/main_tp.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        $fp16_flag \
        --run_id "$run_id" \
        --output_dir ./output \
        --epochs 3 \
        --eval_steps 100 \
        --save_steps 100 \
        --batch_size 64

      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done

done

echo "All experiments completed!"