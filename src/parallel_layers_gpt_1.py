# src/parallel_layers.py

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

class ColumnParallelLinear(nn.Module):
    """
    Splits an nn.Linear's input dimension across world_size GPUs.
    Each rank holds a contiguous shard of the columns of weight.
    """
    def __init__(self, orig_linear: nn.Linear, world_size: int, group: dist.ProcessGroup):
        super().__init__()
        self.group = group
        self.world_size = world_size

        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        assert in_features % world_size == 0, "in_features must be divisible by world_size"
        self.local_in = in_features // world_size

        # allocate local weight shard
        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in, device=torch.cuda.current_device())
        )
        self.bias = orig_linear.bias and nn.Parameter(
            torch.empty(out_features, device=torch.cuda.current_device())
        ) or None

        # split pretrained weight into input shards
        weight_shards = orig_linear.weight.data.chunk(world_size, dim=1)
        rank = dist.get_rank(group=self.group)
        self.weight.data.copy_(weight_shards[rank])

        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_features] -> slice local columns
        x_shard = x[:, dist.get_rank(group=self.group)*self.local_in : (dist.get_rank(group=self.group)+1)*self.local_in]
        local_out = F.linear(x_shard, self.weight, None)  # no local bias

        # all-reduce partial outputs
        dist.all_reduce(local_out, group=self.group)
        if self.bias is not None:
            local_out = local_out + self.bias
        return local_out

class VocabParallelEmbedding(nn.Module):
    """
    Splits nn.Embedding's vocab dimension across world_size GPUs.
    Each rank holds a contiguous slice of the embedding rows.
    """
    def __init__(self, orig_emb: nn.Embedding, world_size: int, group: dist.ProcessGroup):
        super().__init__()
        self.group = group
        self.world_size = world_size

        vocab_size, embed_size = orig_emb.weight.size()
        assert vocab_size % world_size == 0, "vocab_size must be divisible by world_size"
        self.local_vocab = vocab_size // world_size

        self.weight = nn.Parameter(
            torch.empty(self.local_vocab, embed_size, device=torch.cuda.current_device())
        )
        rank = dist.get_rank(group=self.group)
        vocab_shards = orig_emb.weight.data.chunk(world_size, dim=0)
        self.weight.data.copy_(vocab_shards[rank])

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # map global IDs to local range
        rank = dist.get_rank(group=self.group)
        start = rank * self.local_vocab
        mask = (input_ids >= start) & (input_ids < start + self.local_vocab)
        local_ids = input_ids - start
        local_ids = local_ids * mask + torch.zeros_like(local_ids)
        out = F.embedding(local_ids, self.weight)
        # zero out embeddings for IDs outside this shard
        out = out * mask.unsqueeze(-1)
        # all-gather embedding outputs
        gathered = [torch.zeros_like(out) for _ in range(self.world_size)]
        dist.all_gather(gathered, out, group=self.group)
        return torch.cat(gathered, dim=-1)


def parallelize_gpt2(model, world_size=None, group=None):
    """
    Replace GPT-2 modules with tensor-parallel equivalents.
    Assumes `model` is an instance of OpenAI GPT2Model from `src/model.py`.
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if group is None:
        group = dist.group.WORLD
    # 1) Replace token embeddings
    model.wte = VocabParallelEmbedding(model.wte, world_size, group)
    # 2) Transformer blocks
    for block in model.h:
        # QKV combined projection: c_attn -> three columns
        c_attn = block.attn.c_attn
        qkv_size = c_attn.out_features  # 3*hidden
        # split into Q, K, V layers
        q_proj = ColumnParallelLinear(nn.Linear(c_attn.in_features, qkv_size // 3), world_size, group)
        k_proj = ColumnParallelLinear(nn.Linear(c_attn.in_features, qkv_size // 3), world_size, group)
        v_proj = ColumnParallelLinear(nn.Linear(c_attn.in_features, qkv_size // 3), world_size, group)
        block.attn.q_proj = q_proj
        block.attn.k_proj = k_proj
        block.attn.v_proj = v_proj
        # remove original c_attn
        del block.attn.c_attn

        # Output projection
        block.attn.c_proj = RowParallelLinear(block.attn.c_proj, world_size, group)

        # MLP up and down
        block.mlp.c_fc = ColumnParallelLinear(block.mlp.c_fc, world_size, group)
        block.mlp.c_proj = RowParallelLinear(block.mlp.c_proj, world_size, group)

    # 3) Final LM head
    model.lm_head = RowParallelLinear(model.lm_head, world_size, group)
    return model

# Example usage:
# dist.init_process_group("nccl", init_method="env://")\#
# from model import GPT2Model
# model = GPT2Model(config)
# parallelize_gpt2(model)
# model.cuda()
