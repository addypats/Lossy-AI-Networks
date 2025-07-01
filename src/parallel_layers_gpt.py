# src/parallel_layers.py

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

class RowParallelLinear(nn.Module):
    """
    Splits an nn.Linear's output dimension across world_size GPUs.
    Each rank holds a contiguous shard of the rows.
    """
    def __init__(self, orig_linear: nn.Linear, world_size: int, group: dist.ProcessGroup):
        super().__init__()
        self.group = group
        self.world_size = world_size

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
        weight_shards = orig_linear.weight.data.chunk(world_size, dim=0)
        rank = dist.get_rank(group=self.group)
        self.weight.data.copy_(weight_shards[rank])

        # split pretrained bias if exists
        if orig_linear.bias is not None:
            bias_shards = orig_linear.bias.data.chunk(world_size, dim=0)
            self.bias.data.copy_(bias_shards[rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        local_out = F.linear(x, self.weight, self.bias)  # [batch, out/world_size]

        print("####################################################")
        print("\nHello World\n")
        print("####################################################")

        # gather local outputs from all ranks
        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=self.group)
        return torch.cat(gathered, dim=-1)  # [batch, out_features]
    

class ColumnParallelLinear(nn.Module):
    """
    Splits an nn.Linear’s input dimension across world_size GPUs.
    Each rank holds a contiguous shard of the columns.
    """
    def __init__(self, orig_linear: nn.Linear, world_size: int, group: dist.ProcessGroup):
        super().__init__()
        self.group = group
        self.world_size = world_size

        in_f  = orig_linear.in_features
        out_f = orig_linear.out_features
        assert in_f % world_size == 0, "in_features must be divisible by world_size"
        self.local_in = in_f // world_size

        # slice the weight matrix columns
        weight_shards = orig_linear.weight.data.chunk(world_size, dim=1)
        self.weight = nn.Parameter(weight_shards[group.rank()].clone())

        # bias is kept full-size (will be added after reduction)
        self.bias = orig_linear.bias.clone() if orig_linear.bias is not None else None

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: [batch, in_f] -> take only our local columns
        start = self.local_in * self.group.rank()
        end   = start + self.local_in
        x_shard = inputs[:, start:end]                   # [batch, local_in]

        print("####################################################")
        print("\nHello World\n")
        print("####################################################")

        # local partial output
        out_shard = x_shard @ self.weight.t()             # [batch, out_f]

        # gather partials by summing across ranks
        dist.all_reduce(out_shard, group=self.group)      # in-place

        # add bias once
        if self.bias is not None:
            out_shard = out_shard + self.bias

        return out_shard

