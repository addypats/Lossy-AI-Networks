#!/usr/bin/env python3
"""
Megatron-LM + Lossy Network Integration for Fine-tuning
This script provides the simplest integration of your lossy network with Megatron-LM's
tensor parallelism framework for studying loss effects during LLM fine-tuning.
"""

import torch
import torch.distributed as dist
from typing import Optional, Any
import functools

# Import your existing lossy network class
# from your_lossy_module import LossyNetwork

class LossyMegatronIntegration:
    """
    Integrates your lossy network with Megatron-LM's tensor parallel communication.
    This is the recommended approach for studying loss effects during fine-tuning.
    """
    
    def __init__(self, loss_network, enable_on_forward=True, enable_on_backward=True):
        """
        Args:
            loss_network: Your existing lossy network instance
            enable_on_forward: Apply losses to forward pass communications
            enable_on_backward: Apply losses to backward pass communications (gradients)
        """
        self.loss_network = loss_network
        self.enable_on_forward = enable_on_forward
        self.enable_on_backward = enable_on_backward
        self.original_functions = {}
        self.is_enabled = False
        
    def enable(self):
        """Enable lossy communication by monkey-patching Megatron functions."""
        if self.is_enabled:
            return
            
        try:
            # Import Megatron's tensor parallel mappings
            from megatron.core.tensor_parallel import mappings
            
            # Store original functions
            self.original_functions = {
                'reduce_from_tensor_model_parallel_region': mappings._reduce,
                'gather_from_tensor_model_parallel_region': mappings._gather_along_last_dim,
                'reduce_scatter_to_sequence_parallel_region': getattr(mappings, '_reduce_scatter_along_first_dim', None),
                'all_gather_from_sequence_parallel_region': getattr(mappings, '_gather_along_first_dim', None),
            }
            
            # Replace with lossy versions
            mappings._reduce = self._lossy_reduce
            mappings._gather_along_last_dim = self._lossy_gather_along_last_dim
            
            if self.original_functions['reduce_scatter_to_sequence_parallel_region']:
                mappings._reduce_scatter_along_first_dim = self._lossy_reduce_scatter
            if self.original_functions['all_gather_from_sequence_parallel_region']:
                mappings._gather_along_first_dim = self._lossy_gather_along_first_dim
                
            self.is_enabled = True
            print("✅ Lossy Megatron integration enabled")
            
        except ImportError as e:
            print(f"❌ Failed to import Megatron modules: {e}")
            print("Make sure Megatron-LM is installed: pip install megatron-core")
            raise
            
    def disable(self):
        """Restore original Megatron functions."""
        if not self.is_enabled:
            return
            
        try:
            from megatron.core.tensor_parallel import mappings
            
            # Restore original functions
            mappings._reduce = self.original_functions['reduce_from_tensor_model_parallel_region']
            mappings._gather_along_last_dim = self.original_functions['gather_from_tensor_model_parallel_region']
            
            if self.original_functions['reduce_scatter_to_sequence_parallel_region']:
                mappings._reduce_scatter_along_first_dim = self.original_functions['reduce_scatter_to_sequence_parallel_region']
            if self.original_functions['all_gather_from_sequence_parallel_region']:
                mappings._gather_along_first_dim = self.original_functions['all_gather_from_sequence_parallel_region']
                
            self.is_enabled = False
            print("✅ Lossy Megatron integration disabled")
            
        except Exception as e:
            print(f"❌ Error disabling lossy integration: {e}")

    def _apply_lossy_logic(self, tensor, operation_name):
        """Apply your lossy network logic to the tensor."""
        if not hasattr(tensor, 'requires_grad'):
            return tensor
            
        # Determine if this is forward or backward pass
        is_backward = tensor.requires_grad and tensor.grad is not None
        
        # Apply lossy logic based on configuration
        if (is_backward and self.enable_on_backward) or (not is_backward and self.enable_on_forward):
            try:
                # Use your existing lossy network
                if hasattr(self.loss_network, 'send'):
                    # Assuming your lossy network has a 'send' method that returns a mask
                    mask = self.loss_network.send(tensor)
                    if mask is not None:
                        tensor = tensor * mask
                        print(f"🔄 Applied lossy mask in {operation_name}")
                elif hasattr(self.loss_network, 'apply_loss'):
                    # Alternative API
                    tensor = self.loss_network.apply_loss(tensor)
                    print(f"🔄 Applied lossy transformation in {operation_name}")
                    
            except Exception as e:
                print(f"⚠️ Error applying lossy logic in {operation_name}: {e}")
                
        return tensor

    def _lossy_reduce(self, input_tensor, group=None):
        """Lossy version of tensor parallel reduce operation."""
        # Apply lossy logic before reduction
        processed_tensor = self._apply_lossy_logic(input_tensor, "reduce")
        
        # Call original function
        original_fn = self.original_functions['reduce_from_tensor_model_parallel_region']
        return original_fn(processed_tensor, group)

    def _lossy_gather_along_last_dim(self, input_tensor, group=None):
        """Lossy version of tensor parallel gather operation."""
        # Apply lossy logic before gathering
        processed_tensor = self._apply_lossy_logic(input_tensor, "gather_last_dim")
        
        # Call original function
        original_fn = self.original_functions['gather_from_tensor_model_parallel_region']
        return original_fn(processed_tensor, group)

    def _lossy_reduce_scatter(self, input_tensor, group=None):
        """Lossy version of sequence parallel reduce-scatter operation."""
        processed_tensor = self._apply_lossy_logic(input_tensor, "reduce_scatter")
        
        original_fn = self.original_functions['reduce_scatter_to_sequence_parallel_region']
        return original_fn(processed_tensor, group)

    def _lossy_gather_along_first_dim(self, input_tensor, group=None):
        """Lossy version of sequence parallel all-gather operation."""
        processed_tensor = self._apply_lossy_logic(input_tensor, "gather_first_dim")
        
        original_fn = self.original_functions['all_gather_from_sequence_parallel_region']
        return original_fn(processed_tensor, group)


class MegatronLossyFineTuner:
    """
    Main class for fine-tuning with Megatron-LM and lossy networks.
    This provides the complete workflow for your research.
    """
    
    def __init__(self, loss_network, model_config=None):
        """
        Args:
            loss_network: Your lossy network instance
            model_config: Megatron model configuration dict
        """
        self.loss_network = loss_network
        self.model_config = model_config or self._get_default_config()
        self.lossy_integration = LossyMegatronIntegration(loss_network)
        
    def _get_default_config(self):
        """Default configuration for GPT fine-tuning."""
        return {
            'num_layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'seq_length': 1024,
            'max_position_embeddings': 1024,
            'vocab_size': 50257,
            'tensor_model_parallel_size': 2,  # Adjust based on your setup
            'pipeline_model_parallel_size': 1,
        }
    
    def setup_model_provider(self):
        """Create a model provider function for Megatron training."""
        def model_provider(pre_process=True, post_process=True):
            """Model provider function for Megatron."""
            try:
                from megatron.core.models.gpt import GPTModel
                from megatron.core.transformer.spec_utils import import_module
                
                # Create model with your config
                model = GPTModel(
                    config=self.model_config,
                    transformer_layer_spec=None,  # Use default
                    vocab_size=self.model_config['vocab_size'],
                    max_sequence_length=self.model_config['seq_length'],
                    pre_process=pre_process,
                    post_process=post_process,
                )
                
                return model
                
            except ImportError as e:
                print(f"❌ Error importing Megatron model: {e}")
                print("Make sure you have the correct Megatron-LM version installed")
                raise
                
        return model_provider
    
    def run_fine_tuning(self, train_dataset, val_dataset=None, num_epochs=3):
        """
        Run fine-tuning with lossy communication.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            num_epochs: Number of training epochs
        """
        print("🚀 Starting Megatron + Lossy Network Fine-tuning")
        
        # Enable lossy integration
        self.lossy_integration.enable()
        
        try:
            # Setup Megatron training
            from megatron.training import pretrain
            from megatron.training.utils import get_batch_on_this_cp_rank
            
            def train_valid_test_datasets_provider(train_val_test_num_samples):
                """Provide datasets for Megatron training."""
                train_ds, valid_ds, test_ds = None, None, None
                
                if train_dataset is not None:
                    train_ds = train_dataset
                if val_dataset is not None:
                    valid_ds = val_dataset
                    
                return train_ds, valid_ds, test_ds
            
            def forward_step_func(data_iterator, model):
                """Forward step function for Megatron training."""
                batch = get_batch_on_this_cp_rank(data_iterator)
                
                # Forward pass with lossy communication
                output = model(batch['input_ids'], attention_mask=batch.get('attention_mask'))
                
                # Compute loss (adapt based on your task)
                if 'labels' in batch:
                    loss = torch.nn.functional.cross_entropy(
                        output.logits.view(-1, output.logits.size(-1)),
                        batch['labels'].view(-1),
                        ignore_index=-100
                    )
                else:
                    loss = output.loss
                    
                return loss, {'loss': loss}
            
            # Get model provider
            model_provider = self.setup_model_provider()
            
            # Run Megatron training
            pretrain(
                train_valid_test_datasets_provider,
                model_provider,
                forward_step_func,
                args_defaults={
                    'tensor_model_parallel_size': self.model_config['tensor_model_parallel_size'],
                    'pipeline_model_parallel_size': self.model_config['pipeline_model_parallel_size'],
                    'train_iters': num_epochs * 1000,  # Adjust based on dataset size
                    'lr': 1e-5,
                    'min_lr': 1e-6,
                    'lr_decay_style': 'cosine',
                    'weight_decay': 0.01,
                    'clip_grad': 1.0,
                    'fp16': True,  # Use mixed precision
                }
            )
            
        except Exception as e:
            print(f"❌ Error during fine-tuning: {e}")
            raise
        finally:
            # Always disable lossy integration when done
            self.lossy_integration.disable()
            print("✅ Fine-tuning completed")


def main():
    """
    Example usage of the Megatron + Lossy integration.
    Adapt this to your specific lossy network and datasets.
    """
    
    # TODO: Replace with your actual lossy network class
    class MockLossyNetwork:
        """Mock lossy network for demonstration."""
        def __init__(self, loss_rate=0.1):
            self.loss_rate = loss_rate
            
        def send(self, tensor):
            """Apply random dropout as a simple loss simulation."""
            if tensor.requires_grad:
                return torch.bernoulli(torch.full_like(tensor, 1 - self.loss_rate))
            return None
    
    # Initialize your lossy network
    loss_network = MockLossyNetwork(loss_rate=0.05)  # 5% loss rate
    
    # Create fine-tuner
    fine_tuner = MegatronLossyFineTuner(
        loss_network=loss_network,
        model_config={
            'num_layers': 12,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'seq_length': 512,
            'tensor_model_parallel_size': 2,  # Adjust based on your GPU setup
        }
    )
    
    # TODO: Replace with your actual datasets
    print("📝 Note: Replace mock datasets with your actual training data")
    train_dataset = None  # Your training dataset
    val_dataset = None    # Your validation dataset
    
    # Run fine-tuning (uncomment when you have real datasets)
    # fine_tuner.run_fine_tuning(train_dataset, val_dataset, num_epochs=3)
    
    # For now, just test the integration
    integration = LossyMegatronIntegration(loss_network)
    print("🧪 Testing lossy integration setup...")
    
    try:
        integration.enable()
        print("✅ Integration test passed!")
    except ImportError:
        print("⚠️ Megatron-LM not installed. Install with: pip install megatron-core")
    finally:
        integration.disable()


if __name__ == "__main__":
    main()
