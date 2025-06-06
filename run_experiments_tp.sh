#!/bin/bash
# Launch 4 ranks for tensor parallelism (one per GPU)

source /home/ubuntu/tp-env/bin/activate

# # NCCL tuning
# export NCCL_DEBUG=WARN
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


# !–– New NCCL fixes ––!
export NCCL_NET_OFI_DISABLE=1
export NCCL_SOCKET_IFNAME=ens5
export NCCL_IB_DISABLE=1
export NCCL_LAUNCH_TIMEOUT=1200
export NCCL_TIMEOUT=1200

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Torchrun binary
TORCHRUN=$(which torchrun)

# Rendezvous settings
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

# GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Tensor-parallel world size
# TP_SIZE=(2 4)
TP_SIZE=(4)

# Loss-rate grid
LOSS_RATES=(0 0.001 0.005 0.01)
# LOSS_RATES=(0.005 0.01)

# Datasets
# DATASETS=("winogrande" "mnli" "hellaswag" "piqa")
DATASETS=("sst2")

# Precision Flags
# FP_FLAGS=(fp32 fp16)
FP_FLAGS=(fp32)

# Ensure output directory exists
# mkdir -p output_Llama3.2-1B
mkdir -p output_gpt2-large_sst2

for temp_flag in "${FP_FLAGS[@]}"; do
echo
  # Decide on the actual flag to pass into the Python script
  if [ "$temp_flag" = "fp16" ]; then
    fp_flag="--fp16"
    echo
    echo "=== Starting with precision ${fp_flag} ===\n"
    echo
  else
    fp_flag=""   # no flag for fp32
    echo
    echo "=== Starting with precision --fp32 ===\n"
    echo
  fi
  for tp_size in "${TP_SIZE[@]}"; do
  echo
    echo "=== Starting tensor parallelism with size ${tp_size} ==="
    echo
    for dataset in "${DATASETS[@]}"; do
    echo
      echo "=== Starting with dataset ${dataset} ==="
      echo
      for loss_rate in "${LOSS_RATES[@]}"; do
        run_id="tp_gpt2-large_precision-${temp_flag}_Num_Nodes-${tp_size}_lr${loss_rate}"
        echo
        echo "=== Starting $run_id ==="
        echo

        # Original Script
        $TORCHRUN \
          --nproc_per_node $tp_size \
          --master_addr   $MASTER_ADDR \
          --master_port   $MASTER_PORT \
          src/pytorch_train_tp_gpt.py \
            --tensor_parallel_size $tp_size \
            --model_name           "gpt2-large" \
            --dataset              $dataset \
            --batch_size           16 \
            --max_length           128 \
            --learning_rate        3e-5 \
            --weight_decay         0.01 \
            --loss_rate            $loss_rate \
            $fp_flag \
            --seed                 1234 \
            --max_samples          0 \
            --target_accuracy      0.75 \
            --eval_steps           100 \
            --patience             3 \
            --max_steps            100000 \
            --output_dir           "output_gpt2-large_sst2/$run_id" \

        # New set of parameters - mod tp script
        # $TORCHRUN \
        #   --nproc_per_node $tp_size \
        #   --master_addr   $MASTER_ADDR \
        #   --master_port   $MASTER_PORT \
        #   src/Mod_src_TP/pytorch_train_tp.py \
        #     --tensor_parallel_size $tp_size \
        #     --model_name           "meta-llama/Llama-3.2-1B" \
        #     --dataset              $dataset \
        #     --batch_size           8 \
        #     --max_length           256 \
        #     --learning_rate        2e-5 \
        #     --weight_decay         0.01 \
        #     --loss_rate            $loss_rate \
        #     --fp16 \
        #     --seed                 1234 \
        #     --max_samples          0 \
        #     --target_accuracy      0.75 \
        #     --eval_steps           100 \
        #     --patience             5 \
        #     --max_steps            100000 \
        #     --output_dir           "output_Llama3.2-1B/$run_id" \
        #    --run_name $run_id

        echo "=== Completed $run_id ==="
        echo
        echo
      done
    done
  done
done

echo "All tensor-parallel runs done."
