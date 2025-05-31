# Make GPUs 0,1,2,3 visible to all commands in this script
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Now your run loop can refer to those four GPUs without needing to set CUDA_VISIBLE_DEVICES again
MODEL="meta-llama/Llama-3.2-1B"
DATASET="piqa"

# LOSS_RATES=("0.0" "0.001" "0.005" "0.01")
# NUM_NODES=("2" "4")
# PRECISION=("32" "16")

LOSS_RATES=("0.001")
NUM_NODES=("4")
PRECISION=("32")

mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  for nodes in "${NUM_NODES[@]}"; do
    for prec in "${PRECISION[@]}"; do
      run_id="${nodes}gpu_${DATASET}_lr${loss_rate}_fp${prec}"
      echo "Starting experiment: $run_id"

      fp16_flag=""
      if [ "$prec" == "16" ]; then
        fp16_flag="--fp16"
      fi

      python src/src_PP/main.py \
        --model_name "$MODEL" \
        --dataset "$DATASET" \
        --loss_rate "$loss_rate" \
        --num_nodes "$nodes" \
        --batch_size $((16 * ${nodes})) \
        --learning_rate 2e-5 \
        $fp16_flag \
        -nunf 3 \
        --run_id "$run_id" \
        --epochs 4 \
        --max_length 256 \
        --save_steps 100 \
        --logging_steps 10

      echo "Completed experiment: $run_id"
      echo "------------------------------------"
    done
  done
done

echo "All experiments completed!"