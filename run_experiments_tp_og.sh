#!/bin/bash
# Launch 4 ranks for tensor parallelism (one per GPU)

# Original running script for Tensor Parallelism
# Still works. The other one is a modification of this with a little additional things

source /home/ubuntu/tp-env/bin/activate

export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

TORCHRUN=$(which torchrun)

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355
export CUDA_VISIBLE_DEVICES=0,1,2,3

TP_SIZE=4

mkdir -p output

for loss_rate in 0; do
  for prec in 32; do
    run_id="tp_llama_winogrande_lr${loss_rate}_fp${prec}"
    echo "=== Starting $run_id ==="

    torchrun \
      --nproc_per_node $TP_SIZE \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      src/pytorch_train_tp.py \
        --tensor_parallel_size $TP_SIZE \
        --model_name "meta-llama/Llama-3.2-1B" \
        --dataset "winogrande" \
        --batch_size 2 \
	--max_length 128 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --loss_rate $loss_rate \
        --seed 1234 \
        --max_samples 0 \
        --target_accuracy 0.75 \
        --eval_steps 100 \
        --patience 3 \
        --max_steps 100000 \
        --output_dir "output/${run_id}"

    echo "=== Completed $run_id ==="
  done
done

echo "All tensor-parallel runs done."
