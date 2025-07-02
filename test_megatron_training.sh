#!/bin/bash
# Test script for the new Megatron-LM + Lossy Network training script
# This demonstrates how to use the new train_megatron_lossy.py script

# Set up environment (adjust paths as needed)
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=ens5
export NCCL_IB_DISABLE=1
export NCCL_NET_OFI_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Torchrun binary
TORCHRUN=$(which torchrun)

# Rendezvous settings
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

# Test configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1  # For testing with 2 GPUs
TP_SIZE=8
DATASET="mnli"
MODEL_NAME="gpt2-large"

echo "🧪 Testing new Megatron-LM + Lossy Network training script"
echo "==============================================="
echo "TP Size: $TP_SIZE"
echo "Dataset: $DATASET"
echo "Model: $MODEL_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

ge_config_var=("long_1percent")

# Test 1: Bernoulli loss (your standard uniform loss)
echo "🔬 Test 1: Bernoulli (uniform) loss simulation"
run_id="test_megatron_bernoulli_tp${TP_SIZE}"

$TORCHRUN \
  --nproc_per_node $TP_SIZE \
  --master_addr   $MASTER_ADDR \
  --master_port   $MASTER_PORT \
  src/train_megatron_lossy.py \
    --tensor_parallel_size $TP_SIZE \
    --loss_type            ber \
    --model_name           $MODEL_NAME \
    --dataset              $DATASET \
    --batch_size           8 \
    --max_length           128 \
    --learning_rate        3e-5 \
    --weight_decay         0.01 \
    --loss_rate            0.01 \
    --seed                 1234 \
    --max_samples          1000 \
    --target_accuracy      0.75 \
    --eval_steps           10 \
    --patience             5 \
    --max_steps            100 \
    --output_dir           "output/test_$run_id"

echo ""
echo "✅ Test 1 completed"
echo ""

# Test 2: Gilbert-Elliott loss (your bursty loss model)
echo "🔬 Test 2: Gilbert-Elliott (bursty) loss simulation"
run_id="test_megatron_ge_tp${TP_SIZE}"

$TORCHRUN \
  --nproc_per_node $TP_SIZE \
  --master_addr   $MASTER_ADDR \
  --master_port   $MASTER_PORT \
  src/train_megatron_lossy.py \
    --tensor_parallel_size $TP_SIZE \
    --loss_type            g-e \
    --ge_config            $ge_config_var \
    --model_name           $MODEL_NAME \
    --dataset              $DATASET \
    --batch_size           8 \
    --max_length           128 \
    --learning_rate        3e-5 \
    --weight_decay         0.01 \
    --loss_rate            0.001 \
    --seed                 1234 \
    --max_samples          1000 \
    --target_accuracy      0.75 \
    --eval_steps           10 \
    --patience             5 \
    --max_steps            100 \
    --output_dir           "output/test_$run_id"

echo ""
echo "✅ Test 2 completed"
echo ""

echo "🎉 All tests completed!"
echo ""
echo "📊 Results:"
echo "  - Bernoulli test: output/test_test_megatron_bernoulli_tp${TP_SIZE}/"
echo "  - Gilbert-Elliott test: output/test_test_megatron_ge_tp${TP_SIZE}/"
echo ""
echo "💡 To use with your existing experiments, simply replace:"
echo "     src/pytorch_train_tp_gpt.py"
echo "  with:"
echo "     src/train_megatron_lossy.py"
echo "  in your bash scripts."
