#!/usr/bin/env python3
"""
Adapter for integrating your existing lossy network with Megatron-LM.
This script provides a bridge between your current lossy network implementation
and the Megatron-LM tensor parallelism framework.
"""

import torch
import torch.distributed as dist
from typing import Optional, Dict, Any

class LossyNetworkAdapter:
    """
    Adapter class to make your existing lossy network compatible with Megatron-LM.
    This handles the integration details so you can focus on your research.
    """
    
    def __init__(self, your_lossy_network, config: Optional[Dict] = None):
        """
        Args:
            your_lossy_network: Your existing lossy network instance
            config: Configuration dictionary for the adapter
        """
        self.lossy_network = your_lossy_network
        self.config = config or {}
        self.stats = {
            'total_calls': 0,
            'lossy_calls': 0,
            'forward_calls': 0,
            'backward_calls': 0,
        }
        
    def apply_loss(self, tensor: torch.Tensor, context: str = "unknown") -> torch.Tensor:
        """
        Apply your lossy network to a tensor.
        
        Args:
            tensor: Input tensor
            context: Context information (e.g., "forward", "backward", "attention")
            
        Returns:
            Processed tensor with losses applied
        """
        self.stats['total_calls'] += 1
        
        # Detect if this is forward or backward pass
        is_backward = tensor.requires_grad and hasattr(tensor, 'grad_fn') and tensor.grad_fn is not None
        
        if is_backward:
            self.stats['backward_calls'] += 1
        else:
            self.stats['forward_calls'] += 1
        
        # Apply your lossy network
        try:
            # Method 1: If your lossy network has a 'send' method
            if hasattr(self.lossy_network, 'send'):
                mask = self.lossy_network.send(tensor)
                if mask is not None:
                    self.stats['lossy_calls'] += 1
                    return tensor * mask
                    
            # Method 2: If your lossy network has an 'apply' method
            elif hasattr(self.lossy_network, 'apply'):
                result = self.lossy_network.apply(tensor)
                if result is not tensor:  # If modified
                    self.stats['lossy_calls'] += 1
                return result
                
            # Method 3: If your lossy network is callable
            elif callable(self.lossy_network):
                result = self.lossy_network(tensor)
                if result is not tensor:  # If modified
                    self.stats['lossy_calls'] += 1
                return result
                
        except Exception as e:
            print(f"⚠️ Error applying lossy network in {context}: {e}")
            
        return tensor
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about lossy network usage."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


class SimplifiedMegatronIntegration:
    """
    Simplified integration that focuses on the most common tensor parallel operations.
    This is the recommended approach for getting started quickly.
    """
    
    def __init__(self, lossy_adapter: LossyNetworkAdapter):
        """
        Args:
            lossy_adapter: Configured LossyNetworkAdapter instance
        """
        self.adapter = lossy_adapter
        self.is_enabled = False
        self.original_all_reduce = None
        self.original_all_gather = None
        
    def enable(self):
        """Enable lossy communication by patching torch.distributed functions."""
        if self.is_enabled:
            return
            
        # Store original functions
        self.original_all_reduce = dist.all_reduce
        self.original_all_gather = dist.all_gather
        
        # Replace with lossy versions
        dist.all_reduce = self._lossy_all_reduce
        dist.all_gather = self._lossy_all_gather
        
        self.is_enabled = True
        print("✅ Simplified lossy integration enabled")
        
    def disable(self):
        """Restore original functions."""
        if not self.is_enabled:
            return
            
        dist.all_reduce = self.original_all_reduce
        dist.all_gather = self.original_all_gather
        
        self.is_enabled = False
        print("✅ Simplified lossy integration disabled")
        
    def _lossy_all_reduce(self, tensor, *args, **kwargs):
        """Lossy version of all_reduce."""
        # Apply lossy logic before reduction
        processed_tensor = self.adapter.apply_loss(tensor, "all_reduce")
        
        # Call original function
        return self.original_all_reduce(processed_tensor, *args, **kwargs)
    
    def _lossy_all_gather(self, tensor_list, tensor, *args, **kwargs):
        """Lossy version of all_gather."""
        # Apply lossy logic before gathering
        processed_tensor = self.adapter.apply_loss(tensor, "all_gather")
        
        # Call original function
        return self.original_all_gather(tensor_list, processed_tensor, *args, **kwargs)


class MegatronCompatibilityLayer:
    """
    Compatibility layer that works even without Megatron-LM installed.
    Use this for testing and development.
    """
    
    def __init__(self, lossy_adapter: LossyNetworkAdapter):
        self.adapter = lossy_adapter
        self.integration = SimplifiedMegatronIntegration(lossy_adapter)
        
    def test_integration(self, tensor_size=(4, 768)):
        """Test the lossy integration with dummy tensors."""
        print("🧪 Testing lossy integration...")
        
        # Create test tensors
        test_tensor = torch.randn(*tensor_size, requires_grad=True)
        
        # Test without losses
        original_tensor = test_tensor.clone()
        
        # Enable integration
        self.integration.enable()
        
        try:
            # Test lossy application
            processed_tensor = self.adapter.apply_loss(test_tensor, "test")
            
            # Check if losses were applied
            if not torch.equal(original_tensor, processed_tensor):
                print("✅ Losses successfully applied")
            else:
                print("ℹ️ No losses applied (this might be expected)")
                
            # Print statistics
            stats = self.adapter.get_stats()
            print(f"📊 Stats: {stats}")
            
        finally:
            self.integration.disable()
            
        return True
    
    def create_simple_training_loop(self, model, dataloader, num_steps=10):
        """
        Create a simple training loop with lossy communication.
        This demonstrates how to use the integration in practice.
        """
        print("🚀 Starting simple training loop with lossy communication")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # Enable lossy integration
        self.integration.enable()
        
        try:
            for step, batch in enumerate(dataloader):
                if step >= num_steps:
                    break
                    
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # Backward pass (gradients will go through lossy communication)
                loss.backward()
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                if step % 5 == 0:
                    stats = self.adapter.get_stats()
                    print(f"Step {step}: Loss = {loss.item():.4f}, Lossy stats = {stats}")
                    
        finally:
            self.integration.disable()
            
        final_stats = self.adapter.get_stats()
        print(f"🏁 Training completed. Final stats: {final_stats}")


def create_example_usage():
    """
    Example of how to use the adapter with your existing lossy network.
    """
    
    # Example: Your existing lossy network class
    class YourLossyNetwork:
        """Replace this with your actual lossy network."""
        def __init__(self, loss_rate=0.1):
            self.loss_rate = loss_rate
            
        def send(self, tensor):
            """Your existing send method."""
            if tensor.requires_grad:
                # Simple dropout as example
                mask = torch.bernoulli(torch.full_like(tensor, 1 - self.loss_rate))
                return mask
            return None
    
    # Initialize your lossy network
    your_lossy_net = YourLossyNetwork(loss_rate=0.05)
    
    # Create adapter
    adapter = LossyNetworkAdapter(your_lossy_net)
    
    # Create compatibility layer
    compat_layer = MegatronCompatibilityLayer(adapter)
    
    # Test the integration
    compat_layer.test_integration()
    
    print("\n✅ Example completed successfully!")
    print("💡 Replace YourLossyNetwork with your actual implementation")


if __name__ == "__main__":
    print("🔧 Lossy Network Adapter for Megatron-LM")
    print("This script provides adapters for your existing lossy network.\n")
    
    create_example_usage()
