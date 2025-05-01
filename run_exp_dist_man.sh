#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Configuration
# ----------------------------------------
MODEL="meta-llama/Llama-3.2-1B"
DATASET="winogrande"

# Possible values for each run
LOSS_RATES=(0)            # e.g. 0 0.001 0.005 0.01
NUM_NODES=(2 4 6)
PRECISION=(32)            # e.g. 16 32

# Which GPU(s) to expose
export CUDA_VISIBLE_DEVICES=0

# Where to dump results
mkdir -p output

# ----------------------------------------
# Main loop
# ----------------------------------------
for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for prec in "${PRECISION[@]}"; do

      # build a unique run identifier
      run_id="llama_3_2_1b_${DATASET}_lr${loss_rate}_nodes${nodes}_fp${prec}"
      echo "=== Starting experiment: $run_id ==="

      # toggle fp16 if requested
      fp16_flag=""
      if [[ "$prec" == "16" ]]; then
        fp16_flag="--fp16"
      fi

      # invoke your training entrypoint
      python3 -u src/main.py \
        --model_name   "$MODEL" \
        --dataset      "$DATASET" \
        --loss_rate    "$loss_rate" \
        --num_nodes    "$nodes" \
        $fp16_flag \
        --run_id       "$run_id"

      echo "=== Completed experiment: $run_id ==="
      echo
    done
  done
done

echo "All experiments completed!"
