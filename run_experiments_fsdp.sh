#!/bin/bash

pip install --upgrade accelerate

# Avoid CUDA fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="gpt2-medium"
DATASET="winogrande"

# Only launch on up to 4 GPUs on g4dn.12xlarge
LOSS_RATES=("0")
NUM_NODES=("2" "4")
PRECISION=("16")

mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for prec in "${PRECISION[@]}"; do
      run_id="gpt2-medium_bs_${DATASET}_lr${loss_rate}_nodes${nodes}_fp${prec}"
      echo "Starting experiment: $run_id"

      fp16_flag=""
      if [ "$prec" == "16" ]; then
        fp16_flag="--fp16"
      fi

      torchrun --nproc_per_node=$nodes --master_port=12345 src/main_fsdp.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        --batch_size 16 \
        --epochs 3 \
        $fp16_flag \
        --run_id "$run_id"

      echo "Completed experiment: $run_id"
      echo "--------------------------------"
    done
  done
done

echo "All experiments completed!"

