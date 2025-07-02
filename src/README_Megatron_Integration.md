# Megatron-LM + Lossy Network Integration

This directory contains scripts for integrating your lossy network research with Megatron-LM's tensor parallelism framework. This is the **recommended approach** for studying loss effects during LLM fine-tuning at scale.

## 🎯 Why Use Megatron-LM?

- **Production-Ready**: Battle-tested tensor parallelism optimized for large models
- **Scalability**: Handle models that require multiple GPUs (GPT-3 scale and beyond)
- **Efficiency**: Highly optimized CUDA kernels and communication patterns
- **Fine-tuning Support**: Built-in workflows for downstream task fine-tuning
- **Research Focus**: Spend time on your lossy network research, not debugging distributed training

## 📁 Files Overview

| File | Purpose |
|------|---------|
| `megatron_lossy_integration.py` | **Main integration code** - Hooks into Megatron's communication primitives |
| `lossy_adapter.py` | **Adapter layer** - Makes your existing lossy network compatible |
| `setup_megatron_lossy.py` | **Setup script** - Installs dependencies and creates example files |
| `megatron_lossy_training.py` | **Training script** - Complete fine-tuning workflow (created by setup) |

## 🚀 Quick Start

### 1. Run the Setup Script
```bash
python setup_megatron_lossy.py
```
This will:
- Optionally install Megatron-LM and dependencies
- Create example configuration files
- Generate training scripts

### 2. Test the Integration (Without Megatron)
```bash
python lossy_adapter.py
```
This tests your lossy network integration without requiring Megatron-LM.

### 3. Full Integration (With Megatron)
```bash
python megatron_lossy_training.py --help
```

## 🔧 Integration Methods

### Method 1: Simple Adapter (Recommended for Testing)
Use `lossy_adapter.py` to test your integration:

```python
from lossy_adapter import LossyNetworkAdapter, MegatronCompatibilityLayer

# Your existing lossy network
your_lossy_net = YourLossyNetwork()

# Create adapter
adapter = LossyNetworkAdapter(your_lossy_net)

# Test integration
compat_layer = MegatronCompatibilityLayer(adapter)
compat_layer.test_integration()
```

### Method 2: Full Megatron Integration (Recommended for Research)
Use `megatron_lossy_integration.py` for production research:

```python
from megatron_lossy_integration import LossyMegatronIntegration

# Initialize integration
integration = LossyMegatronIntegration(your_lossy_network)

# Enable lossy communication
integration.enable()

# Run your Megatron training
# ... training code ...

# Disable when done
integration.disable()
```

## 🔌 Adapting Your Lossy Network

The integration works with any lossy network that has one of these interfaces:

### Interface 1: `send()` method (Recommended)
```python
class YourLossyNetwork:
    def send(self, tensor):
        # Return a mask or modified tensor
        mask = your_loss_logic(tensor)
        return mask  # or return modified tensor
```

### Interface 2: `apply()` method
```python
class YourLossyNetwork:
    def apply(self, tensor):
        # Return modified tensor
        return your_loss_logic(tensor)
```

### Interface 3: Callable
```python
class YourLossyNetwork:
    def __call__(self, tensor):
        return your_loss_logic(tensor)
```

## ⚙️ Configuration

### Model Configuration
Edit the model config in your training script:

```python
MODEL_CONFIG = {
    'num_layers': 12,           # Transformer layers
    'hidden_size': 768,         # Hidden dimension
    'num_attention_heads': 12,  # Attention heads
    'tensor_model_parallel_size': 2,  # GPUs for tensor parallelism
    'seq_length': 1024,         # Sequence length
}
```

### Lossy Network Configuration
Configure your loss parameters:

```python
LOSSY_CONFIG = {
    'loss_rate': 0.05,          # 5% loss rate
    'burst_length': 10,         # Burst loss duration
    'enable_on_forward': True,  # Apply to forward pass
    'enable_on_backward': True, # Apply to gradients
}
```

## 🖥️ Hardware Requirements

### Minimum Setup
- 2+ NVIDIA GPUs with CUDA capability
- 16GB+ GPU memory per GPU
- CUDA 11.1+ and compatible PyTorch

### Recommended Setup
- 4+ NVIDIA A100/V100 GPUs
- 32GB+ GPU memory per GPU
- NVLink interconnect for optimal communication

## 🏃‍♂️ Running Training

### Single-Node Multi-GPU
```bash
torchrun --nproc_per_node=2 megatron_lossy_training.py \
    --tensor-model-parallel-size 2 \
    --loss-rate 0.05 \
    --model-size gpt2-large
```

### Multi-Node (Advanced)
```bash
# Node 0
torchrun --nnodes=2 --node_rank=0 --master_addr=node0 \
    --nproc_per_node=4 megatron_lossy_training.py \
    --tensor-model-parallel-size 4

# Node 1  
torchrun --nnodes=2 --node_rank=1 --master_addr=node0 \
    --nproc_per_node=4 megatron_lossy_training.py \
    --tensor-model-parallel-size 4
```

## 📊 Research Benefits

### What This Enables
1. **Larger Models**: Study loss effects on models that require tensor parallelism
2. **Real-World Conditions**: Test on production-scale distributed training
3. **Multiple Loss Types**: Apply different loss patterns to different communication operations
4. **Performance Analysis**: Measure impact of losses on both accuracy and training efficiency

### Key Research Questions You Can Answer
- How do communication losses affect fine-tuning convergence?
- Which tensor parallel operations are most sensitive to losses?
- What loss rates can models tolerate while maintaining performance?
- How do burst losses compare to uniform random losses?

## 🐛 Troubleshooting

### Common Issues

**1. Megatron Import Errors**
```bash
# Install Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -e .
```

**2. CUDA Out of Memory**
- Reduce `micro_batch_size`
- Reduce `seq_length`
- Enable gradient checkpointing

**3. Communication Errors**
- Check `CUDA_VISIBLE_DEVICES`
- Verify network connectivity between nodes
- Ensure consistent NCCL versions

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

1. **Use Mixed Precision**: Enable `--fp16` for memory efficiency
2. **Optimize Batch Size**: Balance `micro_batch_size` and `global_batch_size`
3. **Monitor GPU Utilization**: Use `nvidia-smi` to check GPU usage
4. **Profile Communication**: Use NCCL profiling to identify bottlenecks

## 🤝 Contributing

To extend this integration:

1. **Add New Loss Models**: Implement new lossy network classes
2. **Support More Operations**: Hook into additional Megatron communication primitives
3. **Add Metrics**: Implement detailed loss impact measurements
4. **Optimize Performance**: Reduce overhead of lossy transformations

## 📚 References

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Tensor Parallelism Guide](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)

## 📄 License

This integration code follows the same license as your main project.

---

**Next Steps**: 
1. Run `python setup_megatron_lossy.py` to get started
2. Adapt the example to use your specific lossy network
3. Configure for your hardware setup
4. Begin your research on loss effects in distributed LLM training!
