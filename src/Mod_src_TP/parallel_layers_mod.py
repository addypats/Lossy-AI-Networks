import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

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
        with torch.no_grad():
            self.weight.copy_(weight_shards[rank])

        # split pretrained bias if exists
        if orig_linear.bias is not None:
            bias_shards = orig_linear.bias.data.chunk(world_size, dim=0)
            self.bias.copy_(bias_shards[rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        local_out = F.linear(x, self.weight, self.bias)  # [batch, out/world_size]

        # gather local outputs from all ranks
        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=self.group)
        return torch.cat(gathered, dim=-1)  # [batch, out_features]

def replace_linears(module: nn.Module, world_size: int, group: dist.ProcessGroup):
    """
    Recursively replace nn.Linear in `module` with RowParallelLinear
    whenever out_features % world_size == 0 and layer is not inside attention.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            out_f = child.out_features

            # Skip attention Q, K, V layers based on naming heuristics
            if any(qk in name.lower() for qk in ["query", "key", "value", "q_proj", "k_proj", "v_proj"]):
                continue

            if out_f % world_size == 0:
                wrapped = RowParallelLinear(child, world_size, group)
                setattr(module, name, wrapped)
        else:
            replace_linears(child, world_size, group)
