# EXAMPLE: How to modify your run_experiments_tp.sh to use the new Megatron training script
#
# This shows the minimal changes needed to your existing bash script to use
# the new train_megatron_lossy.py script instead of pytorch_train_tp_gpt.py

# OLD (in your current script):
$TORCHRUN \
  --nproc_per_node $tp_size \
  --master_addr   $MASTER_ADDR \
  --master_port   $MASTER_PORT \
  src/pytorch_train_tp_gpt.py \
    --tensor_parallel_size $tp_size \
    --loss_type            g-e \
    --ge_config            $ge_config \
    --model_name           "gpt2-large" \
    --dataset              $dataset \
    --batch_size           8 \
    --max_length           128 \
    --learning_rate        3e-5 \
    --weight_decay         0.01 \
    --loss_rate            0.001 \
    $fp_flag \
    --seed                 1234 \
    --max_samples          0 \
    --target_accuracy      0.75 \
    --eval_steps           20 \
    --patience             15 \
    --max_steps            500 \
    --output_dir           "output_gpt2-large_BurstyLosses_mnli/$run_id"

# NEW (what you would change it to):
$TORCHRUN \
  --nproc_per_node $tp_size \
  --master_addr   $MASTER_ADDR \
  --master_port   $MASTER_PORT \
  src/train_megatron_lossy.py \
    --tensor_parallel_size $tp_size \
    --loss_type            g-e \
    --ge_config            $ge_config \
    --model_name           "gpt2-large" \
    --dataset              $dataset \
    --batch_size           8 \
    --max_length           128 \
    --learning_rate        3e-5 \
    --weight_decay         0.01 \
    --loss_rate            0.001 \
    $fp_flag \
    --seed                 1234 \
    --max_samples          0 \
    --target_accuracy      0.75 \
    --eval_steps           20 \
    --patience             15 \
    --max_steps            500 \
    --output_dir           "output_gpt2-large_BurstyLosses_mnli/$run_id"

# The ONLY change is:
#   src/pytorch_train_tp_gpt.py  →  src/train_megatron_lossy.py
#
# All your existing parameters work exactly the same way!

# Benefits of the new script:
# ✅ Uses Megatron-LM tensor parallelism when available (faster, more efficient)
# ✅ Falls back to your custom tensor parallelism if Megatron isn't available
# ✅ Same exact interface and parameters as your existing script
# ✅ Integrates your Lossy and Gilbert-Elliott network classes
# ✅ Compatible with all your existing bash script configurations
# ✅ Enhanced logging and monitoring with Weights & Biases
# ✅ Better error handling and progress reporting
