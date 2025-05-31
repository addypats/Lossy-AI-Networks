
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
)


def split_tensor_along_last_dim(tensor, num_partitions):
    """Split tensor into equal parts along the last dimension."""
    assert tensor.size(-1) % num_partitions == 0
    return tensor.chunk(num_partitions, dim=-1)


def all_reduce_tensor(tensor, group):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


class TensorParallelLlamaAttention(LlamaAttention):
    def __init__(self, config, layer_idx, world_size, group):
        super().__init__(config, layer_idx)
        self.world_size = world_size
        self.group = group

        hidden_size = config.hidden_size
        local_dim = hidden_size // world_size

        self.q_proj = nn.Linear(hidden_size, local_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, local_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, local_dim, bias=False)
        self.o_proj = nn.Linear(local_dim, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # TODO: Add rotary embeddings if required

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        # All-reduce to gather all shards of output
        attn_output = all_reduce_tensor(attn_output, self.group)
        return self.o_proj(attn_output), attn_probs


class TensorParallelLlamaMLP(LlamaMLP):
    def __init__(self, config, world_size, group):
        super().__init__(config)
        self.world_size = world_size
        self.group = group

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size // world_size

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        intermediate = F.silu(gate) * up

        # All-reduce before down projection
        intermediate = all_reduce_tensor(intermediate, self.group)
        return self.down_proj(intermediate)


class TensorParallelLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, world_size, group):
        super().__init__(config, layer_idx)
        self.self_attn = TensorParallelLlamaAttention(config, layer_idx, world_size, group)
        self.mlp = TensorParallelLlamaMLP(config, world_size, group)


from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig

class TensorParallelLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig, world_size: int, group: torch.distributed.ProcessGroup):
        super().__init__(config)
        self.world_size = world_size
        self.group = group

        # Replace all decoder layers with TensorParallel versions
        self.layers = nn.ModuleList([
            TensorParallelLlamaDecoderLayer(config, layer_idx, world_size, group)
            for layer_idx in range(config.num_hidden_layers)
        ])
