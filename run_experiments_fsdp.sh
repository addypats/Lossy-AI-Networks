#!/bin/bash

MODEL="meta-llama/Llama-3.2-1B"
DATASET="winogrande"
LOSS_RATES=("0")
NUM_NODES=("2" "4" "6")
PRECISION=("32")

mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for prec in "${PRECISION[@]}"; do
      run_id="llama_3_2_1b_${DATASET}_lr${loss_rate}_nodes${nodes}_fp${prec}"
      echo "Starting experiment: $run_id"

      fp16_flag=""
      if [ "$prec" == "16" ]; then
        fp16_flag="--fp16"
      fi

      torchrun --nproc_per_node=$nodes --master_port=12345 src/main.py \
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
