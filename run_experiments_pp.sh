# # Make GPUs 0,1,2,3 visible to all commands in this script
# export CUDA_VISIBLE_DEVICES=0,1,2,3


# # Torchrun binary
# TORCHRUN=$(which torchrun)

# # Now your run loop can refer to those four GPUs without needing to set CUDA_VISIBLE_DEVICES again
# MODEL="meta-llama/Llama-3.2-1B"
# DATASET="winogrande"

# # LOSS_RATES=("0.0" "0.001" "0.005" "0.01")
# # NUM_NODES=("2" "4")
# # PRECISION=("32" "16")

# LOSS_RATES=("0")
# NUM_NODES=("4")
# PRECISION=("32")

# mkdir -p output

# for loss_rate in "${LOSS_RATES[@]}"; do
#   for nodes in "${NUM_NODES[@]}"; do
#     for prec in "${PRECISION[@]}"; do
#       run_id="${nodes}gpu_${DATASET}_lr${loss_rate}_fp${prec}"
#       echo "Starting experiment: $run_id"

#       fp16_flag=""
#       if [ "$prec" == "16" ]; then
#         fp16_flag="--fp16"
#       fi

#       $TORCHRUN \
#           --nproc_per_node $tp_size \
#           --nnodes=1 \
#           --rdzv_id=pipeline_test \
#           --rdzv_backend=c10d \
#           --rdzv_endpoint=127.0.0.1:29500 \
#           python src/src_PP/main.py \
#             --model_name "$MODEL" \
#             --dataset "$DATASET" \
#             --loss_rate "$loss_rate" \
#             --num_nodes "$nodes" \
#             --batch_size $((16 * ${nodes})) \
#             --learning_rate 2e-5 \
#             $fp16_flag \
#             -nunf 3 \
#             --run_id "$run_id" \
#             --epochs 4 \
#             --max_length 256 \
#             --save_steps 100 \
#             --logging_steps 10

#       echo "Completed experiment: $run_id"
#       echo "------------------------------------"
#     done
#   done
# done

# echo "All experiments completed!"



# Set default values
NUM_GPUS=${NUM_GPUS:-4}
LOSS_RATE=${LOSS_RATE:-0.01}
DATASET=${DATASET:-"winogrande"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.2-1B"}
RUN_ID=${RUN_ID:-"pipeline_test_$(date +%Y%m%d_%H%M%S)"}

echo "Launching pipeline parallel training..."
echo "Number of pipeline stages: $NUM_GPUS"
echo "Loss rate: $LOSS_RATE"
echo "Dataset: $DATASET"
echo "Model: $MODEL_NAME"
echo "Run ID: $RUN_ID"

# For single node with multiple GPUs
if [ "$NUM_GPUS" -gt 1 ]; then
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        main.py \
        --num_nodes $NUM_GPUS \
        --loss_rate $LOSS_RATE \
        --dataset $DATASET \
        --model_name $MODEL_NAME \
        --run_id $RUN_ID \
        --batch_size 16 \
        --epochs 3 \
        --learning_rate 3e-5 \
        --fp16 \
        --eval_steps 50 \
        --save_steps 100 \
        --max_length 256 \
        "${@}"
else
    # Single GPU fallback
    python main.py \
        --num_nodes 1 \
        --loss_rate $LOSS_RATE \
        --dataset $DATASET \
        --model_name $MODEL_NAME \
        --run_id $RUN_ID \
        --batch_size 16 \
        --epochs 3 \
        --learning_rate 3e-5 \
        --fp16 \
        --eval_steps 50 \
        --save_steps 100 \
        --max_length 256 \
        "${@}"
fi