from typing import Any, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn


class LossyAllReduceFwdIdentityBwd(torch.autograd.Function):
    """
    Modified version that applies lossy network simulation during all_reduce
    """
    @staticmethod
    def forward(
        ctx: Any, 
        inputs: torch.Tensor, 
        group: Optional[dist.ProcessGroup] = None,
        lossy_network = None
    ) -> torch.Tensor:
        ctx.lossy_network = lossy_network
        inputs = inputs.clone()
        
        # Apply lossy network simulation if provided
        if lossy_network is not None:
            mask = lossy_network.send(inputs)
            inputs = lossy_network.receive(inputs, mask)
        
        dist.all_reduce(inputs, group=group)
        return inputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return grad_outputs, None, None


class LossyIdentityFwdAllReduceBwd(torch.autograd.Function):
    """
    Modified version that applies lossy network simulation during gradient all_reduce
    """
    @staticmethod
    def forward(
        ctx: Any, 
        inputs: torch.Tensor, 
        group: Optional[dist.ProcessGroup] = None,
        lossy_network = None
    ) -> torch.Tensor:
        ctx.group = group
        ctx.lossy_network = lossy_network
        return inputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        grad_outputs = grad_outputs.clone()
        
        # Apply lossy network simulation to gradients
        if ctx.lossy_network is not None:
            mask = ctx.lossy_network.send(grad_outputs)
            grad_outputs = ctx.lossy_network.receive(grad_outputs, mask)
        
        dist.all_reduce(grad_outputs, group=ctx.group)
        return grad_outputs, None, None


class LinearShardedOutputsLossy(nn.Linear):
    """
    Column-parallel linear layer with lossy network simulation
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: dist.ProcessGroup,
        lossy_network = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        sharded_out_features, remainder = divmod(out_features, group.size())
        assert not remainder, "out_features must be divisible by the ProcessGroup size"
        super().__init__(
            in_features=in_features,
            out_features=sharded_out_features,
            device=device,
            dtype=dtype,
        )

        self.group = group
        self.lossy_network = lossy_network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Wrap the unsharded inputs for backwards-pass correctness with lossy simulation
        x = LossyIdentityFwdAllReduceBwd.apply(inputs, self.group, self.lossy_network)
        x = super().forward(x)
        return x


class LinearShardedInputsLossy(nn.Linear):
    """
    Row-parallel linear layer with lossy network simulation
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: dist.ProcessGroup,
        lossy_network = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        sharded_in_features, remainder = divmod(in_features, group.size())
        assert not remainder, "in_features must be divisible by the ProcessGroup size"
        super().__init__(
            in_features=sharded_in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
        )
        self.group = group
        self.lossy_network = lossy_network

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs @ self.weight.T
        # Wrap the mat-mul in an all-reduce with lossy simulation
        x = LossyAllReduceFwdIdentityBwd.apply(x, self.group, self.lossy_network)
        # Crucial: add the bias _after_ the all-reduce
        x = x + self.bias
        return x


class MLTPLossy(nn.Module):
    """
    Tensor Parallel MLP with Lossy Network Simulation
    """
    def __init__(
        self,
        d_model: int,
        group: Optional[dist.ProcessGroup] = None,
        lossy_network = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        # Fallback to the WORLD process group, if None provided
        group = group or dist.group.WORLD

        self.lin_0 = LinearShardedOutputsLossy(
            d_model, 4 * d_model, group=group, lossy_network=lossy_network, 
            device=device, dtype=dtype
        )
        self.act_fn = nn.GELU()
        self.lin_1 = LinearShardedInputsLossy(
            4 * d_model, d_model, group=group, lossy_network=lossy_network,
            device=device, dtype=dtype
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.lin_0(inputs)
        x = self.act_fn(x)
        x = self.lin_1(x)
        return x


def replace_linear_with_tp_lossy(module, group, lossy_network, target_classes=(nn.Linear,)):
    """
    Replace linear layers with tensor parallel versions with lossy network simulation
    """
    from transformers.models.gpt2.modeling_gpt2 import Conv1D
    
    # Skip if this module is already a TensorParallelGPT2Attention or GPT2MLP
    # Check by class name to avoid circular import issues
    if type(module).__name__ in ['TensorParallelGPT2Attention', 'GPT2MLP', 'TensorParallelGPT2MLP']:
        print(f"Skipping {type(module).__name__} - already tensor parallel or handled separately")
        return
    
    for name, child in list(module.named_children()):
        if isinstance(child, target_classes):
            # Handle different layer types
            if isinstance(child, Conv1D):
                # Conv1D stores weight as [in_features, out_features]
                in_features, out_features = child.weight.shape
            elif isinstance(child, nn.Linear):
                # Linear stores normally
                in_features = child.in_features
                out_features = child.out_features
            else:
                print(f"Skipping unknown layer type: {type(child)}")
                continue
            
            # Choose whether to use row or column parallel based on divisibility
            if out_features % group.size() == 0:
                # Use column parallel (output sharding)
                new_layer = LinearShardedOutputsLossy(
                    in_features, out_features, group, lossy_network,
                    device=child.weight.device, dtype=child.weight.dtype
                )
                # Copy weights
                rank = dist.get_rank(group)
                shard_size = out_features // group.size()
                start_idx = rank * shard_size
                end_idx = start_idx + shard_size
                
                if isinstance(child, Conv1D):
                    # Conv1D weight is [in, out], need to transpose for Linear format [out, in]
                    weight_transposed = child.weight.data.transpose(0, 1)  # [out, in]
                    new_layer.weight.data.copy_(weight_transposed[start_idx:end_idx])
                else:
                    new_layer.weight.data.copy_(child.weight.data[start_idx:end_idx])
                    
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data[start_idx:end_idx])
                    
            elif in_features % group.size() == 0:
                # Use row parallel (input sharding)
                new_layer = LinearShardedInputsLossy(
                    in_features, out_features, group, lossy_network,
                    device=child.weight.device, dtype=child.weight.dtype
                )
                # Copy weights
                rank = dist.get_rank(group)
                shard_size = in_features // group.size()
                start_idx = rank * shard_size
                end_idx = start_idx + shard_size
                
                if isinstance(child, Conv1D):
                    # Conv1D weight is [in, out], need to transpose for Linear format [out, in]
                    weight_transposed = child.weight.data.transpose(0, 1)  # [out, in]
                    new_layer.weight.data.copy_(weight_transposed[:, start_idx:end_idx])
                else:
                    new_layer.weight.data.copy_(child.weight.data[:, start_idx:end_idx])
                    
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
            else:
                # Skip layers that can't be parallelized
                print(f"Skipping layer {name}: dimensions ({in_features}, {out_features}) not divisible by group size {group.size()}")
                continue
                
            setattr(module, name, new_layer)
            print(f"Replaced {name} with tensor parallel version: {in_features}x{out_features}")
        else:
            # Don't recursively process children of TensorParallelGPT2Attention or GPT2MLP
            if type(child).__name__ not in ['TensorParallelGPT2Attention', 'GPT2MLP', 'TensorParallelGPT2MLP']:
                replace_linear_with_tp_lossy(child, group, lossy_network, target_classes)


def replace_gpt2_mlp_with_tp_lossy(module, group, lossy_network):
    """
    Replace GPT2 MLP layers with tensor parallel versions.
    GPT2 MLP has:
    - c_fc: Conv1D that expands from hidden_size to intermediate_size (column-parallel)
    - c_proj: Conv1D that contracts from intermediate_size to hidden_size (row-parallel)
    """
    from transformers.models.gpt2.modeling_gpt2 import Conv1D
    
    for name, child in list(module.named_children()):
        # Check if this is a GPT2MLP module
        if type(child).__name__ == 'GPT2MLP':
            print(f"Found GPT2MLP at {name}, replacing with tensor parallel version")
            
            # Get the original layers
            c_fc = child.c_fc
            c_proj = child.c_proj
            act = child.act
            dropout = child.dropout
            
            # c_fc: Conv1D(nf=intermediate_size, nx=hidden_size) - column parallel
            hidden_size, intermediate_size = c_fc.weight.shape  # Conv1D stores as [nx, nf]
            print(f"c_fc dimensions: {hidden_size} -> {intermediate_size}")
            
            # Create column-parallel c_fc (shard outputs)
            new_c_fc = LinearShardedOutputsLossy(
                hidden_size, intermediate_size, group, lossy_network,
                device=c_fc.weight.device, dtype=c_fc.weight.dtype
            )
            
            # Copy c_fc weights
            rank = dist.get_rank(group)
            shard_size = intermediate_size // group.size()
            start_idx = rank * shard_size
            end_idx = start_idx + shard_size
            
            # Conv1D weight is [nx, nf], need to transpose for Linear format [nf, nx]
            weight_transposed = c_fc.weight.data.transpose(0, 1)  # [nf, nx] = [intermediate, hidden]
            new_c_fc.weight.data.copy_(weight_transposed[start_idx:end_idx])
            if c_fc.bias is not None:
                new_c_fc.bias.data.copy_(c_fc.bias.data[start_idx:end_idx])
            
            # Create row-parallel c_proj (shard inputs)
            new_c_proj = LinearShardedInputsLossy(
                intermediate_size, hidden_size, group, lossy_network,
                device=c_proj.weight.device, dtype=c_proj.weight.dtype
            )
            
            # Copy c_proj weights
            # Conv1D weight is [nx, nf], need to transpose for Linear format [nf, nx]
            weight_transposed = c_proj.weight.data.transpose(0, 1)  # [nf, nx] = [hidden, intermediate]
            new_c_proj.weight.data.copy_(weight_transposed[:, start_idx:end_idx])
            if c_proj.bias is not None:
                new_c_proj.bias.data.copy_(c_proj.bias.data)
                
            # Create a new MLP module with tensor parallel layers
            class TensorParallelGPT2MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.c_fc = new_c_fc
                    self.c_proj = new_c_proj
                    self.act = act
                    self.dropout = dropout
                    
                def forward(self, x):
                    x = self.c_fc(x)
                    x = self.act(x)
                    x = self.c_proj(x)
                    x = self.dropout(x)
                    return x
            
            # Replace the MLP
            setattr(module, name, TensorParallelGPT2MLP())
            print(f"Replaced GPT2MLP {name} with tensor parallel version")
            
        else:
            # Recursively process children, but skip TensorParallelGPT2Attention
            if type(child).__name__ != 'TensorParallelGPT2Attention':
                replace_gpt2_mlp_with_tp_lossy(child, group, lossy_network)
