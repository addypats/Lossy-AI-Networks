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

        # allocate local shard
        self.weight = nn.Parameter(
            torch.empty(self.local_out, in_features, device=torch.cuda.current_device())
        )
        self.bias = nn.Parameter(
            torch.empty(self.local_out, device=torch.cuda.current_device())
        )

        # scatter pretrained weights/bias into shards
        shards_w = orig_linear.weight.data.chunk(world_size, dim=0)
        shards_b = orig_linear.bias.data.chunk(world_size, dim=0)
        rank = dist.get_rank(group=self.group)
        self.weight.data.copy_(shards_w[rank])
        self.bias.data.copy_(shards_b[rank])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features]
        local_out = F.linear(x, self.weight, self.bias)  # [batch, out/world_size]

        # gather local outputs from all ranks
        gathered = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_out, group=self.group)
        return torch.cat(gathered, dim=-1)  # [batch, out_features]
