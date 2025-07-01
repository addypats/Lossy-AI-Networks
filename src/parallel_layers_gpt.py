# # src/parallel_layers.py

# import torch
# import torch.nn as nn
# import torch.distributed as dist
# import torch.nn.functional as F
# from torch import Tensor

# class RowParallelLinear(nn.Module):
#     """
#     Splits an nn.Linear's output dimension across world_size GPUs.
#     Each rank holds a contiguous shard of the rows.
#     """
#     def __init__(self, orig_linear: nn.Linear, world_size: int, group: dist.ProcessGroup):
#         super().__init__()
#         self.group = group
#         self.world_size = world_size

#         in_features = orig_linear.in_features
#         out_features = orig_linear.out_features
#         assert out_features % world_size == 0, "out_features must be divisible by world_size"
#         self.local_out = out_features // world_size

#         # allocate local shard for weight
#         self.weight = nn.Parameter(
#             torch.empty(self.local_out, in_features, device=torch.cuda.current_device())
#         )

#         # handle bias possibly being None
#         if orig_linear.bias is not None:
#             self.bias = nn.Parameter(
#                 torch.empty(self.local_out, device=torch.cuda.current_device())
#             )
#         else:
#             self.bias = None

#         # split pretrained weight into shards
#         weight_shards = orig_linear.weight.data.chunk(world_size, dim=0)
#         rank = dist.get_rank(group=self.group)
#         self.weight.data.copy_(weight_shards[rank])

#         # split pretrained bias if exists
#         if orig_linear.bias is not None:
#             bias_shards = orig_linear.bias.data.chunk(world_size, dim=0)
#             self.bias.data.copy_(bias_shards[rank])

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [batch, in_features]
#         local_out = F.linear(x, self.weight, self.bias)  # [batch, out/world_size]

#         print("####################################################")
#         print("\nHello World\n")
#         print("####################################################")

#         # gather local outputs from all ranks
#         gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
#         dist.all_gather(gathered, local_out, group=self.group)
#         return torch.cat(gathered, dim=-1)  # [batch, out_features]
    

# class ColumnParallelLinear(nn.Module):
#     """
#     Splits an nn.Linear’s input dimension across world_size GPUs.
#     Each rank holds a contiguous shard of the columns.
#     """
#     def __init__(self, orig_linear: nn.Linear, world_size: int, group: dist.ProcessGroup):
#         super().__init__()
#         self.group = group
#         self.world_size = world_size

#         in_f  = orig_linear.in_features
#         out_f = orig_linear.out_features
#         assert in_f % world_size == 0, "in_features must be divisible by world_size"
#         self.local_in = in_f // world_size

#         # slice the weight matrix columns
#         weight_shards = orig_linear.weight.data.chunk(world_size, dim=1)
#         self.weight = nn.Parameter(weight_shards[group.rank()].clone())

#         # bias is kept full-size (will be added after reduction)
#         self.bias = orig_linear.bias.clone() if orig_linear.bias is not None else None

#     def forward(self, inputs: Tensor) -> Tensor:
#         # inputs: [batch, in_f] -> take only our local columns
#         start = self.local_in * self.group.rank()
#         end   = start + self.local_in
#         x_shard = inputs[:, start:end]                   # [batch, local_in]

#         print("####################################################")
#         print("\nHello World\n")
#         print("####################################################")

#         # local partial output
#         out_shard = x_shard @ self.weight.t()             # [batch, out_f]

#         # gather partials by summing across ranks
#         dist.all_reduce(out_shard, group=self.group)      # in-place

#         # add bias once
#         if self.bias is not None:
#             out_shard = out_shard + self.bias

#         return out_shard


import torch
import torch.nn as nn
import torch.distributed as dist
from transformers.models.gpt2.modeling_gpt2 import Conv1D
from torch import Tensor

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer. 
    Input is replicated across all ranks, weights are column-partitioned.
    Output is summed across ranks via all-reduce.
    """
    def __init__(self, orig, world_size, group):
        super().__init__()
        self.group = group
        self.world_size = world_size

        # pull out raw weight & bias
        W = orig.weight.data
        B = orig.bias.data if orig.bias is not None else None

        # unify to [out, in]
        if isinstance(orig, Conv1D):
            W = W.transpose(0,1)
        # now W: [out_f, in_f]

        out_f, in_f = W.shape
        assert in_f % world_size == 0, f"Input features {in_f} not divisible by world_size {world_size}"
        self.in_features = in_f
        self.out_features = out_f
        self.local_in = in_f // world_size

        # shard the columns of W → weight_shard: [out_f, local_in]
        weight_shards = W.chunk(world_size, dim=1)
        rank = dist.get_rank(self.group)
        self.weight = nn.Parameter(weight_shards[rank].clone())

        # bias stays full-size (only rank 0 should have it to avoid duplication)
        if B is not None and rank == 0:
            self.bias = nn.Parameter(B.clone())
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Column parallel: input is replicated, each rank has partial weights
        # x should have shape [batch, in_features] and be identical across all ranks
        
        rank = dist.get_rank(self.group)
        
        # Slice input to match our weight partition
        start = self.local_in * rank
        end = start + self.local_in
        x_local = x[:, start:end]  # [batch, local_in]

        # Compute partial output
        out = torch.matmul(x_local, self.weight.t())  # [batch, out_f]

        # All-reduce to sum partial results
        dist.all_reduce(out, group=self.group)

        # Add bias only on rank 0 to avoid duplication
        if self.bias is not None:
            out = out + self.bias
            
        return out


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer.
    Input is replicated across all ranks, weights are row-partitioned.
    Output is gathered from all ranks via all-gather.
    """
    def __init__(self, orig, world_size, group):
        super().__init__()
        self.group = group
        self.world_size = world_size

        W = orig.weight.data
        B = orig.bias.data if orig.bias is not None else None

        if isinstance(orig, Conv1D):
            # conv stores [in,out], so transpose to [out,in]
            W = W.transpose(0,1)

        # now W: [out_f, in_f]
        out_f, in_f = W.shape
        assert out_f % world_size == 0, f"Output features {out_f} not divisible by world_size {world_size}"
        self.in_features = in_f
        self.out_features = out_f
        self.local_out = out_f // world_size

        # shard the rows → weight_shard: [local_out, in_f]
        weight_shards = W.chunk(world_size, dim=0)
        rank = dist.get_rank(self.group)
        self.weight = nn.Parameter(weight_shards[rank].clone())
        
        # shard bias if it exists
        if B is not None:
            bias_shards = B.chunk(world_size, dim=0)
            self.bias = nn.Parameter(bias_shards[rank].clone())
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Row parallel: input is replicated, each rank computes partial output
        # x should have shape [batch, in_features] and be identical across all ranks
        
        # Validate input shape
        expected_shape = (x.size(0), self.in_features)
        if x.shape != expected_shape:
            raise RuntimeError(f"RowParallel input shape mismatch: got {x.shape}, expected {expected_shape}")
        
        # Each rank computes its output shard with full input
        out_shard = torch.nn.functional.linear(x, self.weight, self.bias)  # [batch, local_out]

        # Gather all shards from all ranks - this is the critical communication step
        output_list = [torch.zeros_like(out_shard) for _ in range(self.world_size)]
        
        try:
            dist.all_gather(output_list, out_shard, group=self.group)
        except Exception as e:
            print(f"Rank {dist.get_rank(self.group)}: all_gather failed with {e}")
            raise
        
        # Concatenate along the output dimension
        result = torch.cat(output_list, dim=-1)  # [batch, out_f]
        return result
