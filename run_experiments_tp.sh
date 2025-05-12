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
TP_SIZE=4

# Loss-rate grid
LOSS_RATES=(0.0001)

# Ensure output directory exists
mkdir -p output

for loss_rate in "${LOSS_RATES[@]}"; do
  run_id="tp_gpt2_winogrande_lr${loss_rate}"
  echo "=== Starting $run_id ==="

  $TORCHRUN \
    --nproc_per_node $TP_SIZE \
    --master_addr   $MASTER_ADDR \
    --master_port   $MASTER_PORT \
    src/pytorch_train_tp.py \
      --tensor_parallel_size $TP_SIZE \
      --model_name           "gpt2-medium" \
      --dataset              "winogrande" \
      --batch_size           2 \
      --max_length           128 \
      --learning_rate        3e-5 \
      --weight_decay         0.01 \
      --loss_rate            $loss_rate \
      --seed                 1234 \
      --max_samples          0 \
      --target_accuracy      0.75 \
      --eval_steps           100 \
      --patience             3 \
      --max_steps            100000 \
      --output_dir           "output/${run_id}"

  echo "=== Completed $run_id ==="
  echo
done

echo "All tensor-parallel runs done."
