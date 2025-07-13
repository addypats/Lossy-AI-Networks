# src/parallel_layers.py

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from transformers.models.gpt2.modeling_gpt2 import Conv1D

class RowParallelLinear(nn.Module):
    """
    Splits an nn.Linear's output dimension across world_size GPUs.
    Each rank holds a contiguous shard of the rows.
    """
    def __init__(self, orig_linear, world_size: int, group: dist.ProcessGroup):
        super().__init__()
        self.group = group
        self.world_size = world_size

        # Handle both nn.Linear and Conv1D layers
        if isinstance(orig_linear, Conv1D):
            # Conv1D.weight is [in_features, out_features]
            in_features, out_features = orig_linear.weight.shape
        else:
            # nn.Linear.weight is [out_features, in_features]
            in_features = orig_linear.in_features
            out_features = orig_linear.out_features
            
        assert out_features % world_size == 0, "out_features must be divisible by world_size"
        self.local_out = out_features // world_size

        # allocate local shard for weight
        self.weight = nn.Parameter(
            torch.empty(self.local_out, in_features, device=torch.cuda.current_device())
        )

        # handle bias possibly being None
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(
                torch.empty(self.local_out, device=torch.cuda.current_device())
            )
        else:
            self.bias = None

        # split pretrained weight into shards
        if isinstance(orig_linear, Conv1D):
            # Conv1D.weight is [in_features, out_features], need to transpose for splitting
            weight_to_split = orig_linear.weight.data.t()  # [out_features, in_features]
        else:
            # nn.Linear.weight is already [out_features, in_features]
            weight_to_split = orig_linear.weight.data
            
        weight_shards = weight_to_split.chunk(world_size, dim=0)
        rank = dist.get_rank(group=self.group)
        self.weight.data.copy_(weight_shards[rank])

        # split pretrained bias if exists
        if orig_linear.bias is not None:
            bias_shards = orig_linear.bias.data.chunk(world_size, dim=0)
            self.bias.data.copy_(bias_shards[rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        local_out = F.linear(x, self.weight, self.bias)  # [batch, out/world_size]

        # gather local outputs from all ranks
        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=self.group)
        return torch.cat(gathered, dim=-1)  # [batch, out_features]
    

# class ColumnParallelLinear(nn.Module):
#     """
#     Splits an nn.Linear’s input dimension across world_size GPUs.
#     Each rank holds a contiguous shard of the columns.
#     """
#     def __init__(self, orig_linear, world_size: int, group: dist.ProcessGroup):
#         super().__init__()
#         self.group = group
#         self.world_size = world_size
#         self._is_conv1d = isinstance(orig_linear, Conv1D)

#         # Handle both nn.Linear and Conv1D layers
#         if isinstance(orig_linear, Conv1D):
#             # Conv1D.weight is [in_features, out_features]
#             in_f, out_f = orig_linear.weight.shape
#         else:
#             # nn.Linear.weight is [out_features, in_features]
#             in_f = orig_linear.in_features
#             out_f = orig_linear.out_features
            
#         assert in_f % world_size == 0, "in_features must be divisible by world_size"
#         self.local_in = in_f // world_size

#         # slice the weight matrix columns
#         if isinstance(orig_linear, Conv1D):
#             # Conv1D.weight is [in_features, out_features], so we chunk along dim 0
#             weight_shards = orig_linear.weight.data.chunk(world_size, dim=0)
#             # No need to transpose - Conv1D uses weight as [in, out] directly
#             self.weight = nn.Parameter(weight_shards[dist.get_rank(group=self.group)].clone())
#         else:
#             # nn.Linear.weight is [out_features, in_features], so we chunk along dim 1
#             weight_shards = orig_linear.weight.data.chunk(world_size, dim=1)
#             self.weight = nn.Parameter(weight_shards[dist.get_rank(group=self.group)].clone())

#         # bias is kept full-size (will be added after reduction)
#         self.bias = orig_linear.bias.clone() if orig_linear.bias is not None else None

#     def forward(self, inputs: Tensor) -> Tensor:
#         # inputs: [batch, in_f] -> take only our local columns
#         start = self.local_in * dist.get_rank(group=self.group)
#         end   = start + self.local_in
#         x_shard = inputs[:, start:end]                   # [batch, local_in]

#         # local partial output - handle different weight formats
#         if hasattr(self, '_is_conv1d') and self._is_conv1d:
#             # For Conv1D, weight is [local_in, out_f]
#             out_shard = x_shard @ self.weight              # [batch, out_f]
#         else:
#             # For nn.Linear, weight is [out_f, local_in]
#             out_shard = x_shard @ self.weight.t()          # [batch, out_f]

#         # gather partials by summing across ranks
#         dist.all_reduce(out_shard, group=self.group)      # in-place

#         # add bias once
#         if self.bias is not None:
#             out_shard = out_shard + self.bias

#         return out_shard

