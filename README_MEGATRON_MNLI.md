# Megatron GPT2-Large MNLI Fine-tuning

This directory contains scripts for fine-tuning GPT2-large on the MNLI dataset using the original Megatron framework with tensor parallelism, W&B logging, and convergence based on running average accuracy.

## Requirements

### Software Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+
- Megatron-LM (NVIDIA's official implementation)
- Weights & Biases account

### Hardware Requirements
- At least 2 GPUs with 16GB+ VRAM each (for tensor parallelism)
- Recommended: 4x A100 or V100 GPUs

### Python Dependencies
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install transformers datasets sklearn wandb tqdm pyyaml
```

## Setup

1. **Clone Megatron-LM**:
   ```bash
   cd "c:\Users\adity\Desktop\Personal Projects\Lossy AI Networks"
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   pip install -e .
   ```

2. **Setup Weights & Biases**:
   ```bash
   wandb login
   ```

3. **Configure environment**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on available GPUs
   export WANDB_PROJECT="megatron-gpt2-mnli-finetune"
   ```

## Usage

### Option 1: Complete Training Script (Recommended)

Run the comprehensive training script that handles data preparation, training, and monitoring:

```bash
chmod +x run_megatron_gpt2_mnli.sh
./run_megatron_gpt2_mnli.sh
```

This script will:
- Download required vocabulary files
- Prepare MNLI data in the correct format
- Start Megatron training with proper GPT2-large configuration
- Monitor training progress and log to W&B
- Evaluate every 20 steps using the last 5 evaluations for convergence
- Stop when running average accuracy ≥ 75%

### Option 2: Manual Steps

1. **Prepare Data**:
   ```bash
   python -c "
   from datasets import load_dataset
   import json
   dataset = load_dataset('glue', 'mnli')
   # Data preparation logic here
   "
   ```

2. **Start Training**:
   ```bash
   python -m torch.distributed.launch \
       --nproc_per_node=2 \
       --nnodes=1 \
       Megatron-LM/pretrain_gpt.py \
       --tensor-model-parallel-size 2 \
       --num-layers 36 \
       --hidden-size 1280 \
       --num-attention-heads 20 \
       --seq-length 512 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --lr 1e-5 \
       --train-iters 10000 \
       --save ./checkpoints \
       --fp16
   ```

3. **Monitor and Evaluate**:
   ```bash
   python evaluate_mnli.py --monitor --checkpoint-dir ./checkpoints
   ```

### Option 3: Evaluation Only

To evaluate existing checkpoints:

```bash
# Monitor training directory
python evaluate_mnli.py --monitor --checkpoint-dir ./checkpoints

# Evaluate single checkpoint
python evaluate_mnli.py --single --checkpoint-path ./checkpoints/iter_0001000

# Run baseline evaluation
python evaluate_mnli.py --baseline
```

## Configuration

### Key Parameters

- **Target Accuracy**: 75% (configurable via `--target-accuracy`)
- **Evaluation Interval**: Every 20 steps (configurable via `--eval-interval`)
- **Convergence Criteria**: Running average of last 5 evaluations ≥ target accuracy
- **Tensor Parallelism**: 2 GPUs (configurable via `--tensor-parallel-size`)
- **Batch Size**: 4 micro-batch per GPU, 32 global batch size
- **Learning Rate**: 1e-5 (fine-tuning rate)
- **Sequence Length**: 512 tokens

### Model Architecture (GPT2-Large)
- **Layers**: 36
- **Hidden Size**: 1280
- **Attention Heads**: 20
- **Parameters**: ~774M
- **Vocabulary Size**: 50,257

## Monitoring and Logging

### Weights & Biases Integration
The scripts automatically log:
- Training loss
- Evaluation accuracy (every 20 steps)
- Running average accuracy (last 5 evaluations)
- Best accuracy achieved
- Learning rate schedule
- Training time and step count

### Console Output
```
Step    20 | Acc: 0.4521 | Avg: 0.4251 | Loss: 1.234567 | LR: 1.00e-05 | Time: 120.5s
Step    40 | Acc: 0.4892 | Avg: 0.4687 | Loss: 1.198432 | LR: 9.98e-06 | Time: 241.2s
...
🎉 TARGET ACHIEVED! 🎉
Running average 0.7523 >= target 0.7500
```

### Checkpoint Management
- Checkpoints saved every 100 steps (configurable)
- Best model saved when accuracy improves
- Final model saved when target accuracy reached
- Evaluation results saved as JSON

## File Structure

```
Lossy AI Networks/
├── run_megatron_gpt2_mnli.sh          # Main training script
├── evaluate_mnli.py                    # Evaluation and monitoring
├── megatron_wrapper.py                 # Training wrapper with W&B
├── megatron_gpt2_finetune.py          # Full Megatron integration
├── megatron_mnli_simple.py            # Simplified version
├── Megatron-LM/                       # Cloned Megatron repository
├── checkpoints/                       # Training checkpoints
├── data/                              # Prepared datasets
│   ├── gpt2-vocab.json
│   ├── gpt2-merges.txt
│   ├── mnli_train.jsonl
│   └── mnli_valid.jsonl
└── src/
    └── config.yaml                    # Dataset configurations
```

## Troubleshooting

### Common Issues

1. **GPU Memory Errors**:
   - Reduce `--micro-batch-size` from 4 to 2 or 1
   - Reduce `--seq-length` from 512 to 256
   - Enable gradient checkpointing: `--checkpoint-activations`

2. **Import Errors**:
   - Ensure Megatron-LM is properly installed: `pip install -e ./Megatron-LM`
   - Check Python path: `export PYTHONPATH=./Megatron-LM:$PYTHONPATH`

3. **Data Loading Issues**:
   - Verify MNLI data is downloaded: `python -c "from datasets import load_dataset; load_dataset('glue', 'mnli')"`
   - Check data preparation script output

4. **Distributed Training Issues**:
   - Verify NCCL installation: `python -c "import torch; print(torch.distributed.is_nccl_available())"`
   - Check GPU visibility: `nvidia-smi`
   - Ensure consistent CUDA versions

### Performance Optimization

1. **For faster training**:
   - Use FP16: `--fp16` (already enabled)
   - Increase batch size if memory allows
   - Use multiple nodes if available

2. **For memory efficiency**:
   - Enable activation checkpointing
   - Use gradient accumulation
   - Reduce sequence length for initial testing

## Expected Results

### Training Timeline
- **Steps 0-500**: Initial accuracy ~33% (random performance)
- **Steps 500-2000**: Gradual improvement to ~50-60%
- **Steps 2000-5000**: Steady improvement to ~65-70%
- **Steps 5000+**: Fine-tuning to target 75%+ accuracy

### Convergence
- Target: Running average of 5 evaluations ≥ 75%
- Expected convergence: 3000-7000 steps
- Total training time: 2-6 hours (depending on hardware)

## Notes

- The scripts include both actual Megatron integration and mock evaluation for testing
- W&B logging can be disabled with `--no-wandb` flag
- Checkpoints are compatible with standard Megatron format
- The evaluation uses MNLI validation-matched split for consistency
- No packet loss simulation (as requested for "original" Megatron performance)

## Support

For issues specific to:
- **Megatron-LM**: Check [NVIDIA Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- **MNLI Dataset**: Check [GLUE benchmark documentation](https://gluebenchmark.com/)
- **Weights & Biases**: Check [W&B documentation](https://docs.wandb.ai/)
