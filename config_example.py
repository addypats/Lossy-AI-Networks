
# Example Megatron Configuration for Lossy Network Research
# Adjust these parameters based on your hardware and research needs

MODEL_CONFIG = {
    # Model architecture
    'num_layers': 12,           # Number of transformer layers
    'hidden_size': 768,         # Hidden dimension
    'num_attention_heads': 12,  # Number of attention heads
    'seq_length': 1024,         # Sequence length
    'max_position_embeddings': 1024,
    'vocab_size': 50257,        # GPT-2 vocabulary size
    
    # Parallelism settings
    'tensor_model_parallel_size': 2,   # Number of GPUs for tensor parallelism
    'pipeline_model_parallel_size': 1, # Number of GPUs for pipeline parallelism
    
    # Training settings
    'micro_batch_size': 4,      # Micro batch size per GPU
    'global_batch_size': 32,    # Total batch size across all GPUs
    'lr': 1e-5,                # Learning rate
    'min_lr': 1e-6,            # Minimum learning rate
    'weight_decay': 0.01,       # Weight decay
    'clip_grad': 1.0,          # Gradient clipping
    
    # Mixed precision
    'fp16': True,              # Use FP16 mixed precision
    'bf16': False,             # Use BF16 (if supported)
    
    # Checkpointing
    'save_interval': 1000,      # Save checkpoint every N iterations
    'eval_interval': 500,       # Evaluate every N iterations
}

# Lossy Network Configuration
LOSSY_CONFIG = {
    'loss_rate': 0.05,          # 5% packet loss rate
    'burst_length': 10,         # Burst loss length
    'enable_on_forward': True,  # Apply losses to forward pass
    'enable_on_backward': True, # Apply losses to backward pass (gradients)
}

# Dataset Configuration
DATASET_CONFIG = {
    'train_data_path': 'path/to/your/train/data',
    'valid_data_path': 'path/to/your/valid/data',
    'tokenizer_name': 'gpt2',
    'seq_length': 1024,
    'split': '949,50,1',  # Train, validation, test split
}
