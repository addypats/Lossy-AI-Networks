
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributed as dist
# from transformers.models.llama.modeling_llama import (
#     LlamaMLP,
#     LlamaAttention,
#     LlamaDecoderLayer,
# )


# def split_tensor_along_last_dim(tensor, num_partitions):
#     """Split tensor into equal parts along the last dimension."""
#     assert tensor.size(-1) % num_partitions == 0
#     return tensor.chunk(num_partitions, dim=-1)


# def all_reduce_tensor(tensor, group):
#     dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
#     return tensor


# class TensorParallelLlamaAttention(LlamaAttention):
#     def __init__(self, config, layer_idx, world_size, group):
#         super().__init__(config, layer_idx)
#         self.world_size = world_size
#         self.group = group

#         hidden_size = config.hidden_size
#         local_dim = hidden_size // world_size

#         self.q_proj = nn.Linear(hidden_size, local_dim, bias=False)
#         self.k_proj = nn.Linear(hidden_size, local_dim, bias=False)
#         self.v_proj = nn.Linear(hidden_size, local_dim, bias=False)
#         self.o_proj = nn.Linear(local_dim, hidden_size, bias=False)

#     def forward(self, hidden_states, **kwargs):
#         q = self.q_proj(hidden_states)
#         k = self.k_proj(hidden_states)
#         v = self.v_proj(hidden_states)

#         # TODO: Add rotary embeddings if required

#         attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
#         attn_probs = F.softmax(attn_weights, dim=-1)
#         attn_output = torch.matmul(attn_probs, v)

#         # All-reduce to gather all shards of output
#         attn_output = all_reduce_tensor(attn_output, self.group)
#         return self.o_proj(attn_output), attn_probs


# class TensorParallelLlamaMLP(LlamaMLP):
#     def __init__(self, config, world_size, group):
#         super().__init__(config)
#         self.world_size = world_size
#         self.group = group

#         hidden_size = config.hidden_size
#         intermediate_size = config.intermediate_size // world_size

#         self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
#         self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

#     def forward(self, hidden_states):
#         gate = self.gate_proj(hidden_states)
#         up = self.up_proj(hidden_states)
#         intermediate = F.silu(gate) * up

#         # All-reduce before down projection
#         intermediate = all_reduce_tensor(intermediate, self.group)
#         return self.down_proj(intermediate)


# class TensorParallelLlamaDecoderLayer(LlamaDecoderLayer):
#     def __init__(self, config, layer_idx, world_size, group):
#         super().__init__(config, layer_idx)
#         self.self_attn = TensorParallelLlamaAttention(config, layer_idx, world_size, group)
#         self.mlp = TensorParallelLlamaMLP(config, world_size, group)


# from transformers.models.llama.modeling_llama import LlamaModel, LlamaConfig

# class TensorParallelLlamaModel(LlamaModel):
#     def __init__(self, config: LlamaConfig, world_size: int, group: torch.distributed.ProcessGroup):
#         super().__init__(config)
#         self.world_size = world_size
#         self.group = group

#         # Replace all decoder layers with TensorParallel versions
#         self.layers = nn.ModuleList([
#             TensorParallelLlamaDecoderLayer(config, layer_idx, world_size, group)
#             for layer_idx in range(config.num_hidden_layers)
#         ])




import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaRMSNorm,       # LayerNorm in Llama is RMSNorm
    LlamaRotaryEmbedding,
)

# --------------------------------------------------------------------------------
# Utility functions for weight sharding and collective ops
# --------------------------------------------------------------------------------

def split_parameter(tensor: torch.Tensor, dim: int, world_size: int):
    """
    Splits `tensor` into `world_size` equal chunks along `dim`, returns a list of chunks.
    """
    assert tensor.size(dim) % world_size == 0, f"Dimension {dim}={tensor.size(dim)} not divisible by world_size={world_size}"
    return torch.chunk(tensor, world_size, dim=dim)


def all_gather_tensor(tensor: torch.Tensor, group: dist.ProcessGroup):
    """
    All-gather a tensor along its last dimension. Returns a concatenated full tensor on each rank.
    """
    world_size = dist.get_world_size(group)
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)
    return torch.cat(tensor_list, dim=-1)


def all_reduce_tensor(tensor: torch.Tensor, group: dist.ProcessGroup):
    """
    All-reduce (SUM) a tensor in-place across `group`.
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


# --------------------------------------------------------------------------------
# 1) Row‐Parallel Linear
#    Splits a full Linear(out_features × in_features) weight by rows.
#    Forward: local matmul → [batch, <local_out>] → all-gather → [batch, out_features].
#    Backward: gradient is automatically computed on each shard; no further reduce needed.
# --------------------------------------------------------------------------------

class RowParallelLinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, world_size: int, group: torch.distributed.ProcessGroup):
        """
        Creates a row-parallel wrapper around a pretrained nn.Linear.
        The original Linear has shape [out_f, in_f]. We assert out_f % world_size == 0.

        self.weight (Parameter): shape [out_f/world_size, in_f]
        self.bias   (Parameter, optional): shape [out_f/world_size]
        """
        super().__init__()
        self.world_size = world_size
        self.group = group

        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        assert out_features % world_size == 0, \
            f"RowParallel: out_features ({out_features}) not divisible by world_size ({world_size})"
        self.local_out = out_features // world_size

        # Allocate a local weight shard
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features, device=orig_linear.weight.device))
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(torch.empty(self.local_out, device=orig_linear.weight.device))
        else:
            self.bias = None

        # We do NOT initialize here; pretrained copying happens externally
        self._orig_out = out_features
        self._orig_in  = in_features

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, in_features]
        Returns: [batch_size, out_features] on each rank
        """
        # 1) local matmul → [batch_size, local_out]
        local_output = F.linear(x, self.weight, self.bias)  # [batch, local_out]

        # 2) all-gather local_output along hidden dim → [batch, out_features]
        output_full = all_gather_tensor(local_output, self.group)
        return output_full


# --------------------------------------------------------------------------------
# 2) Column‐Parallel Linear
#    Splits a full Linear(out_features × in_features) weight by columns (in_features dimension).
#    Forward: all-gather input slice? Actually column-parallel means input has been gathered already
#    so that each rank sees [batch, in_f/world_size] → matmul with [out_f, in_f/world_size] → [batch, out_f]
#    Then all-reduce across ranks to sum partial results.
# --------------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, world_size: int, group: torch.distributed.ProcessGroup):
        """
        Creates a column-parallel wrapper around a pretrained nn.Linear.
        Original Linear: weight shape [out_f, in_f]
        We assert in_f % world_size == 0.
        Each rank gets a local weight [out_f, in_f/world_size].

        Forward:
          - Gather input features from all ranks to reconstruct full [batch, in_f] OR
            assume that the caller has already gathered full x. Here, we'll assume x is full [batch, in_f].
          - Multiply: x[..., local_in_feats] @ local_weight.T   (→ [batch, out_f])
          - All-reduce the partial [batch, out_f] across ranks → final [batch, out_f].
        """
        super().__init__()
        self.world_size = world_size
        self.group = group

        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        assert in_features % world_size == 0, \
            f"ColumnParallel: in_features ({in_features}) not divisible by world_size ({world_size})"
        self.local_in = in_features // world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in, device=orig_linear.weight.device))
        if orig_linear.bias is not None:
            # Bias is shared, so we keep a full-size bias on each rank
            self.bias = nn.Parameter(torch.empty(out_features, device=orig_linear.weight.device))
        else:
            self.bias = None

        self._orig_out = out_features
        self._orig_in  = in_features

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, orig_in_features]
        We assume the caller has already stacked/gathered x across ranks if needed (i.e. x is full).
        Each rank will use x[:, rank*local_in : (rank+1)*local_in] to do its part of matmul.

        Returns: [batch_size, out_features] (after all-reduce across ranks).
        """
        # 1) Slice input columns for this rank
        rank = dist.get_rank(self.group)
        start = rank * self.local_in
        end = start + self.local_in
        x_local = x[:, start:end]  # [batch, local_in]

        # 2) Local matmul: [batch, out_f] = x_local [batch, local_in] @ weight.T [local_in, out_f]
        local_output = F.linear(x_local, self.weight.t(), None)  # [batch, out_features]

        # 3) All-reduce (SUM) partial outputs → final full output [batch, out_f]
        output_full = all_reduce_tensor(local_output, self.group)
        if self.bias is not None:
            output_full = output_full + self.bias
        return output_full


# --------------------------------------------------------------------------------
# 3) TensorParallelLlamaAttention
#    We take a pretrained fused QKV weight ([H, 3H]) and split it row-wise among world_size:
#      - Each rank holds [H, 3H/W] chunk → F.linear → [batch, seq, 3H/W] → chunk into (Q, K, V) each [H/W].
#    We then:
#      - All-gather K_local, V_local to reconstruct K_full, V_full [H].
#      - Compute attention on Q_local vs K_full, V_full → partial output [H/W].
#      - Row-parallel “O”: each rank has a local slice of the O weight ([H/W, H]) → multiply local_input [H] → [H/W] → all-gather → [H].
#    We also include attention_mask, rotary embeddings placeholder, residual + LayerNorm + dropout.
# --------------------------------------------------------------------------------

class TensorParallelLlamaAttention(nn.Module):
    def __init__(self,
                 llama_config: LlamaConfig,
                 layer_idx: int,
                 world_size: int,
                 group: torch.distributed.ProcessGroup,
                 pretrained_qkv_weight: torch.Tensor,
                 pretrained_qkv_bias: torch.Tensor,
                 pretrained_o_weight: torch.Tensor,
                 pretrained_o_bias: torch.Tensor):
        """
        llama_config: HuggingFace LlamaConfig
        pretrained_qkv_weight: full [H, 3H] weight from pretrained qkv_proj
        pretrained_qkv_bias:   full [3H] bias from pretrained qkv_proj (or None)
        pretrained_o_weight:   full [H, H] weight from pretrained o_proj
        pretrained_o_bias:     full [H] bias from pretrained o_proj (or None)
        """
        # super().__init__()
        # super().__init__(
        #     llama_config,
        #     layer_idx,
        #     use_cache=False,
        #     num_key_value_heads=llama_config.num_key_value_heads,
        #     head_dim=llama_config.hidden_size // llama_config.num_attention_heads,
        #     # (no rotary_embedding arguments)
        # )
        
        super().__init__(llama_config, layer_idx)
        
        self.world_size = world_size
        self.group = group
        self.hidden_size = llama_config.hidden_size
        self.num_heads = llama_config.num_attention_heads
        self.head_dim  = self.hidden_size // self.num_heads
        assert self.hidden_size % world_size == 0, "hidden_size must be divisible by world_size"
        self.local_hidden = self.hidden_size // world_size   # H/W

        # rotary embeddings
        # self.rotary_emb = LlamaRotaryEmbedding(
        #     base=llama_config.rotary_embedding_base,
        #     head_dim=self.head_dim,
        #     freqs_for="vertical"
        # )

        # LayerNorm before attention
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=llama_config.rms_norm_eps)

        # 1) Row-parallel QKV projection: splits fused [H, 3H] → [H, 3H/W]
        #    Each rank’s qkv weight shard is one chunk of the full weight (dim=1 split).
        self.qkv_weight = nn.Parameter(torch.empty(self.hidden_size, (3 * self.hidden_size) // world_size,
                                                   device=pretrained_qkv_weight.device))
        if pretrained_qkv_bias is not None:
            self.qkv_bias = nn.Parameter(torch.empty((3 * self.hidden_size) // world_size,
                                                      device=pretrained_qkv_bias.device))
        else:
            self.qkv_bias = None

        # 2) Column-parallel O projection: each rank holds a slice [H/W, H]
        self.o_weight = nn.Parameter(torch.empty(self.local_hidden, self.hidden_size,
                                                 device=pretrained_o_weight.device))
        if pretrained_o_bias is not None:
            # In row-parallel O, bias is shared fully on each rank
            self.o_bias = nn.Parameter(torch.empty(self.hidden_size, device=pretrained_o_bias.device))
        else:
            self.o_bias = None

        # Dropout layers
        self.attn_dropout = nn.Dropout(llama_config.attention_dropout)
        self.resid_dropout = nn.Dropout(llama_config.hidden_dropout)

        # Load pretrained shards into our parameters
        # -----------------------------------------------------------
        #  a) Split full qkv weight [H, 3H] → world_size chunks along dim=1
        qkv_chunks = split_parameter(pretrained_qkv_weight, dim=1, world_size=world_size)
        rank = dist.get_rank(self.group)
        self.qkv_weight.data.copy_(qkv_chunks[rank])  # [H, 3H/W]
        if pretrained_qkv_bias is not None:
            bias_chunks = split_parameter(pretrained_qkv_bias, dim=0, world_size=world_size)
            self.qkv_bias.data.copy_(bias_chunks[rank])

        #  b) Split full o_proj weight [H, H] → world_size chunks along dim=0 (rows)
        o_weight_chunks = split_parameter(pretrained_o_weight, dim=0, world_size=world_size)
        self.o_weight.data.copy_(o_weight_chunks[rank])  # [H/W, H]
        if pretrained_o_bias is not None:
            # Bias is full [H], shared on each rank
            self.o_bias.data.copy_(pretrained_o_bias)

        # Cache shapes
        self._orig_qkv_out = 3 * self.hidden_size
        self._orig_o_out   = self.hidden_size

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                **kwargs):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, 1, 1, seq_len] (0 = mask, 1 = keep)
        position_ids:   [batch_size, seq_len] if used with rotary

        Returns:
          new_hidden_states: [batch_size, seq_len, hidden_size]
          attn_probs:       [batch_size, num_heads, seq_len, seq_len]  (on each rank, identical)
        """
        # 1) Pre-LN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 2) Fused QKV projection (row-parallel)
        #    local_qkv: [batch, seq, 3*(H/W)]
        #    bias broadcast: [3H/W]
        local_qkv = F.linear(hidden_states,
                             self.qkv_weight, 
                             self.qkv_bias)  # [batch, seq, 3(H/W)]

        # 3) Chunk into q_local, k_local, v_local each [batch, seq, H/W]
        q_local, k_local, v_local = torch.chunk(local_qkv, 3, dim=-1)

        # 4) Reshape [batch, seq, H/W] → [batch, num_heads, seq, head_dim/W?] carefully
        #    First view: [batch, seq, H/W] → [batch, seq, num_heads, head_dim * (1/W)]
        #    Actually, split H/W across all heads: H/W = (num_heads * head_dim)/W.
        new_shape = (q_local.size(0), q_local.size(1), self.num_heads, self.head_dim // self.world_size)
        q_local = q_local.view(*new_shape).transpose(1, 2)  # → [batch, num_heads, seq, head_dim/W]
        k_local = k_local.view(*new_shape).transpose(1, 2)
        v_local = v_local.view(*new_shape).transpose(1, 2)

        # 5) Apply rotary embeddings to Q_local & K_local
        #    (LlamaRotaryEmbedding returns [batch, num_heads, seq, head_dim/W])
        if position_ids is not None:
            q_local, k_local = self.rotary_emb(q_local, k_local, position_ids)

        # 6) All-gather K_local and V_local across ranks → full [batch, num_heads, seq, head_dim]
        #    Note: rank 0 … W-1 each has [batch, num_heads, seq, head_dim/W]
        k_gathered = all_gather_tensor(k_local, self.group)  # [batch, num_heads, seq, head_dim]
        v_gathered = all_gather_tensor(v_local, self.group)  # [batch, num_heads, seq, head_dim]

        # 7) Compute attention scores: [batch, num_heads, seq, seq]
        #    (q_local: [batch, num_heads, seq, head_dim/W], k_gathered: [batch, num_heads, seq, head_dim])
        #    → cast k_gathered back to [batch, num_heads, head_dim, seq] for matmul
        #    We do q_local @ k_gathered.transpose(-1,-2)
        attn_scores = torch.matmul(
            q_local,
            k_gathered.transpose(-1, -2)
        ) / ( (self.head_dim) ** 0.5 )  # scaled by sqrt(d_head)

        # 8) Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, 1, 1, seq] or [batch, 1, seq, seq]
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)  # [batch, num_heads, seq, seq]
        attn_probs = self.attn_dropout(attn_probs)

        # 9) Compute attention output: attn_probs @ V_full → [batch, num_heads, seq, head_dim]
        attn_output_local = torch.matmul(attn_probs, v_gathered)  # → [batch, num_heads, seq, head_dim]

        # 10) Merge heads: → [batch, seq, H/W]
        attn_output_local = attn_output_local.transpose(1, 2).contiguous() \
                            .view(attn_output_local.size(0), attn_output_local.size(2), self.local_hidden)

        # 11) Row-parallel O projection:
        #     Each rank’s local o_weight: [H/W, H]
        #     We need the full [batch, seq, H] as input → so all-gather attn_output_local → [batch, seq, H]
        attn_output_full = all_gather_tensor(attn_output_local, self.group)  # [batch, seq, H]

        # 12) Multiply by local O weight: [H/W, H] → [batch, seq, H/W], then gather those→[batch, seq, H]
        local_o_out = F.linear(attn_output_full, self.o_weight, None)  # [batch, seq, H/W]

        # 13) All-gather row-parallel O outputs
        o_out_full = all_gather_tensor(local_o_out, self.group)  # [batch, seq, H]
        if self.o_bias is not None:
            o_out_full = o_out_full + self.o_bias

        # 14) Apply residual + dropout
        o_out_full = self.resid_dropout(o_out_full)
        o_out_full = residual + o_out_full  # skip connection

        return o_out_full, attn_probs


# --------------------------------------------------------------------------------
# 4) TensorParallelLlamaMLP
#
#    We take pretrained up_proj ([H, 4H]) and down_proj ([4H, H]).
#    - Row-parallel “up_proj”: each rank holds [H, (4H)/W] chunk → local matmul → [batch, seq, 4H/W].
#    - All-gather intermediate shards → [batch, seq, 4H]
#    - Column-parallel “down_proj”: each rank holds [ (4H)/W, H ] → slice input → local matmul → [batch, seq, H]
#    - All-reduce output → final [batch, seq, H]
#
#    We also include: LayerNorm → MLP → dropout → residual.
# --------------------------------------------------------------------------------

class TensorParallelLlamaMLP(nn.Module):
    def __init__(self,
                 llama_config: LlamaConfig,
                 world_size: int,
                 group: torch.distributed.ProcessGroup,
                 pretrained_up_weight: torch.Tensor,
                 pretrained_up_bias: torch.Tensor,
                 pretrained_down_weight: torch.Tensor,
                 pretrained_down_bias: torch.Tensor):
        """
        llama_config: HuggingFace LlamaConfig
        pretrained_up_weight:   full [H, 4H]
        pretrained_up_bias:     full [4H] or None
        pretrained_down_weight: full [4H, H]
        pretrained_down_bias:   full [H] or None
        """
        super().__init__()
        self.world_size = world_size
        self.group = group
        self.hidden_size = llama_config.hidden_size
        self.intermediate_size = llama_config.intermediate_size  # = 4H
        assert self.intermediate_size == 4 * self.hidden_size

        # LayerNorm post-attn
        self.post_attn_layernorm = LlamaRMSNorm(self.hidden_size, eps=llama_config.rms_norm_eps)

        # Row-parallel “up”:
        #   Each rank’s up_weight shard: [H, intermediate_size/W] = [H, (4H)/W].
        self.up_weight = nn.Parameter(torch.empty(self.hidden_size, self.intermediate_size // world_size,
                                                  device=pretrained_up_weight.device))
        if pretrained_up_bias is not None:
            self.up_bias = nn.Parameter(torch.empty(self.intermediate_size // world_size,
                                                    device=pretrained_up_bias.device))
        else:
            self.up_bias = None

        # Column-parallel “down”:
        #   Each rank’s down_weight shard: [ (4H)/W, H ]
        self.down_weight = nn.Parameter(torch.empty(self.intermediate_size // world_size, self.hidden_size,
                                                    device=pretrained_down_weight.device))
        if pretrained_down_bias is not None:
            # This full bias [H] is shared on each rank
            self.down_bias = nn.Parameter(torch.empty(self.hidden_size, device=pretrained_down_bias.device))
        else:
            self.down_bias = None

        # Dropout
        self.resid_dropout = nn.Dropout(llama_config.hidden_dropout)

        # # # Pretrained weight splitting # # #
        rank = dist.get_rank(self.group)

        # 1) Split up_weight [H, 4H] → world_size chunks along dim=1
        up_weight_chunks = split_parameter(pretrained_up_weight, dim=1, world_size=world_size)
        self.up_weight.data.copy_(up_weight_chunks[rank])  # [H, 4H/W]
        if pretrained_up_bias is not None:
            up_bias_chunks = split_parameter(pretrained_up_bias, dim=0, world_size=world_size)
            self.up_bias.data.copy_(up_bias_chunks[rank])

        # 2) Split down_weight [4H, H] → world_size chunks along dim=0
        down_weight_chunks = split_parameter(pretrained_down_weight, dim=0, world_size=world_size)
        self.down_weight.data.copy_(down_weight_chunks[rank])  # [4H/W, H]
        if pretrained_down_bias is not None:
            # Bias is full [H], shared on each rank
            self.down_bias.data.copy_(pretrained_down_bias)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch, seq, H]
        Returns: [batch, seq, H]
        """
        # 1) Pre‐MLP LayerNorm
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)

        # 2) Row-parallel “up”: project H → 4H/W on each rank
        #    local_intermediate: [batch, seq, 4H/W]
        local_intermediate = F.linear(hidden_states, self.up_weight, self.up_bias)

        # 3) Apply activation (SiLU) on local_intermediate
        local_intermediate = F.silu(local_intermediate)

        # 4) All-gather local_intermediate → [batch, seq, 4H] on every rank
        intermediate_full = all_gather_tensor(local_intermediate, self.group)

        # 5) Column-parallel “down”: each rank’s down_weight: [4H/W, H]
        #    We need to slice intermediate_full along last dim:
        rank = dist.get_rank(self.group)
        start = rank * (self.intermediate_size // self.world_size)
        end   = start + (self.intermediate_size // self.world_size)
        intermediate_local_slice = intermediate_full[:, :, start:end]  # [batch, seq, 4H/W]

        # 6) Local down matmul: [batch, seq, H]
        local_mlp_output = F.linear(intermediate_local_slice, self.down_weight, None)  # → [batch, seq, H]

        # 7) All-reduce across ranks → sum partials → final [batch, seq, H]
        mlp_output_full = all_reduce_tensor(local_mlp_output, self.group)
        if self.down_bias is not None:
            mlp_output_full = mlp_output_full + self.down_bias

        # 8) Dropout + Residual
        mlp_output_full = self.resid_dropout(mlp_output_full)
        mlp_output_full = residual + mlp_output_full

        return mlp_output_full


# --------------------------------------------------------------------------------
# 5) TensorParallelLlamaDecoderLayer
#    Replaces a single decoder block in the original Llama model.
#    It chains:
#      - Attention (parallel) → MLP (parallel), each preceded by the correct LayerNorm / followed by dropout + residual.
# --------------------------------------------------------------------------------

from transformers.models.llama.modeling_llama import LlamaDecoderLayer as _HF_LlamaDecoderLayer

class TensorParallelLlamaDecoderLayer(nn.Module):
    def __init__(self,
                 llama_config: LlamaConfig,
                 layer_idx: int,
                 world_size: int,
                 group: torch.distributed.ProcessGroup,
                 hf_decoder_layer: _HF_LlamaDecoderLayer):
        """
        llama_config: the config of the model
        layer_idx: index of this layer in the stack
        world_size, group: for distributed
        hf_decoder_layer: the HuggingFace LlamaDecoderLayer instance, from which we extract pretrained weights
        """
        super().__init__()
        self.world_size = world_size
        self.group = group
        self.config = llama_config

        # Extract pretrained weights from hf_decoder_layer
        
        # qkv_w = hf_decoder_layer.self_attn.q_proj.weight.data    # [H, H]
        # qkv_b = hf_decoder_layer.self_attn.q_proj.bias.data      # [H]
        
        # Note: HuggingFace’s Llama fuses Q, K, V into a single tensor by concatenation if use_fused? 
        # Actually in code they do q_proj = nn.Linear(H, H) and k_proj, v_proj separately. 
        # But in many versions they instead store as a single fused weight. We must check carefully:
        # The current HuggingFace `LlamaAttention` has:
        #   self.q_proj = nn.Linear(H, H); self.k_proj = nn.Linear(H, H); self.v_proj = nn.Linear(H, H)
        # That means each is [H, H]. They do not fuse them in one matmul. 
        # But our TP implementation expects a fused [H, 3H]. We can construct it manually:
        #    fused_qkv = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=1), similarly for bias.

        # # 1) Build fused QKV weight & bias:
        # q_w = hf_decoder_layer.self_attn.q_proj.weight.data           # [H, H]
        # k_w = hf_decoder_layer.self_attn.k_proj.weight.data           # [H, H]
        # v_w = hf_decoder_layer.self_attn.v_proj.weight.data           # [H, H]
        # fused_qkv_w = torch.cat([q_w, k_w, v_w], dim=1)               # [H, 3H]

        # # If biases exist:
        # if hf_decoder_layer.self_attn.q_proj.bias is not None:
          #   q_b = hf_decoder_layer.self_attn.q_proj.bias.data         # [H]
          #   k_b = hf_decoder_layer.self_attn.k_proj.bias.data         # [H]
          #   v_b = hf_decoder_layer.self_attn.v_proj.bias.data         # [H]
          #   fused_qkv_b = torch.cat([q_b, k_b, v_b], dim=0)           # [3H]
        # else:
          #   fused_qkv_b = None

        # 1) Build fused QKV weight & bias in one of three possible layouts:
        #    (a) a single `qkv_proj` of shape [H, 3H]
        #    (b) three independent [H, H] q_proj/k_proj/v_proj
        #    (c) grouped K/V: q_proj is [H, H], but k_proj/v_proj are [H, Dk] with Dk < H,
        #        so we tile K/V to full [H, H].

        H = llama_config.hidden_size

        # Case (a): fused qkv_proj exists
        if hasattr(hf_decoder_layer.self_attn, "qkv_proj"):
            raw_qkv = hf_decoder_layer.self_attn.qkv_proj.weight.data
            if raw_qkv.shape == (H, 3 * H):
                fused_qkv_w = raw_qkv.clone()
                if (hf_decoder_layer.self_attn.qkv_proj.bias is not None
                    and hf_decoder_layer.self_attn.qkv_proj.bias.data.shape[0] == 3 * H):
                    fused_qkv_b = hf_decoder_layer.self_attn.qkv_proj.bias.data.clone()
                else:
                    fused_qkv_b = None
            else:
                raise RuntimeError(
                    f"Found fused qkv_proj, but its weight is {raw_qkv.shape}, expected ({H}, {3*H})"
                )

        # Case (b) or (c): separate q_proj/k_proj/v_proj exist
        elif hasattr(hf_decoder_layer.self_attn, "q_proj") \
            and hasattr(hf_decoder_layer.self_attn, "k_proj") \
            and hasattr(hf_decoder_layer.self_attn, "v_proj"):

            # Grab raw weight tensors
            raw_q_w = hf_decoder_layer.self_attn.q_proj.weight.data   # could be [H, H]
            raw_k_w = hf_decoder_layer.self_attn.k_proj.weight.data   # could be [Dk, H] or [H, Dk]
            raw_v_w = hf_decoder_layer.self_attn.v_proj.weight.data   # could be [Dv, H] or [H, Dv]

            # 1) Ensure Q is [H, H]
            if raw_q_w.shape != (H, H):
                raise RuntimeError(
                    f"Unexpected q_proj shape {raw_q_w.shape}, expected [H, H]."
                )
            q_w = raw_q_w  # shape [H, H]

            # 2) If K is [Dk, H], transpose → [H, Dk]. Otherwise, if it is [H, Dk], use directly.
            if raw_k_w.shape[0] == H:
                # K already [H, Dk]
                k_w = raw_k_w
            elif raw_k_w.shape[1] == H:
                # K stored as [Dk, H], so transpose to [H, Dk]
                k_w = raw_k_w.transpose(0, 1)
            else:
                raise RuntimeError(
                    f"k_proj has shape {raw_k_w.shape}, which is neither [H, Dk] nor [Dk, H]."
                )

            # 3) Likewise for V
            if raw_v_w.shape[0] == H:
                v_w = raw_v_w
            elif raw_v_w.shape[1] == H:
                v_w = raw_v_w.transpose(0, 1)
            else:
                raise RuntimeError(
                    f"v_proj has shape {raw_v_w.shape}, which is neither [H, Dv] nor [Dv, H]."
                )

            # 4) Now we have q_w: [H, H]; k_w: [H, Dk]; v_w: [H, Dv].
            Dk = k_w.shape[1]
            Dv = v_w.shape[1]

            # If all three dims are [H, H], concatenate directly
            if (q_w.shape == (H, H)) and (Dk == H) and (Dv == H):
                fused_qkv_w = torch.cat([q_w, k_w, v_w], dim=1)  # [H, 3H]
            else:
                # Must tile K and V so that their widths become H exactly
                if (q_w.shape != (H, H)):
                    raise RuntimeError(
                        f"Unexpected q_proj shape {q_w.shape}, expected [H, H]."
                    )

                # Check Dk and Dv divide H
                if (H % Dk != 0) or (H % Dv != 0):
                    raise RuntimeError(
                        f"Grouped K/V widths must divide H. H={H}, Dk={Dk}, Dv={Dv}."
                    )

                reps_k = H // Dk
                reps_v = H // Dv
                k_w_full = k_w.repeat_interleave(reps_k, dim=1)  # [H, H]
                v_w_full = v_w.repeat_interleave(reps_v, dim=1)  # [H, H]
                fused_qkv_w = torch.cat([q_w, k_w_full, v_w_full], dim=1)  # [H, 3H]

            # 5) Handle biases similarly
            raw_q_b = hf_decoder_layer.self_attn.q_proj.bias.data if hf_decoder_layer.self_attn.q_proj.bias is not None else None
            raw_k_b = hf_decoder_layer.self_attn.k_proj.bias.data if hf_decoder_layer.self_attn.k_proj.bias is not None else None
            raw_v_b = hf_decoder_layer.self_attn.v_proj.bias.data if hf_decoder_layer.self_attn.v_proj.bias is not None else None

            if raw_q_b is not None and raw_k_b is not None and raw_v_b is not None:
                # q_b must be [H]
                if raw_q_b.shape != (H,):
                    fused_qkv_b = None
                else:
                    # k_b: could be [H] or [Dk]
                    if raw_k_b.shape == (H,):
                        k_b_full = raw_k_b
                    elif raw_k_b.shape == (Dk,):
                        if H % Dk != 0:
                            k_b_full = None
                        else:
                            k_b_full = raw_k_b.repeat_interleave(H // Dk)
                    else:
                        k_b_full = None

                    # v_b: could be [H] or [Dv]
                    if raw_v_b.shape == (H,):
                        v_b_full = raw_v_b
                    elif raw_v_b.shape == (Dv,):
                        if H % Dv != 0:
                            v_b_full = None
                        else:
                            v_b_full = raw_v_b.repeat_interleave(H // Dv)
                    else:
                        v_b_full = None

                    # If all three bias vectors are valid length H, concatenate
                    if k_b_full is not None and v_b_full is not None:
                        fused_qkv_b = torch.cat([raw_q_b, k_b_full, v_b_full], dim=0)  # [3H]
                    else:
                        fused_qkv_b = None
            else:
                fused_qkv_b = None

        else:
            # No valid QKV found
            all_names = [name for name, _ in hf_decoder_layer.self_attn.named_parameters()]
            raise RuntimeError(
                "Cannot find a valid QKV in this LlamaAttention. "
                "self_attn parameters = "
                + ", ".join(all_names)
            )  

        # 2) O projection weight & bias:
        o_w = hf_decoder_layer.self_attn.o_proj.weight.data           # [H, H]
        o_b = hf_decoder_layer.self_attn.o_proj.bias.data if hf_decoder_layer.self_attn.o_proj.bias is not None else None

        # 3) MLP up/down:
        # up_w   = hf_decoder_layer.mlp.gate_proj.weight.data          # [H, 4H]
        # up_b   = hf_decoder_layer.mlp.gate_proj.bias.data            # [4H]  (SiLU gate bias)
        up_w = hf_decoder_layer.mlp.gate_proj.weight.data             # [H, 4H]
        if hf_decoder_layer.mlp.gate_proj.bias is not None:
            up_b = hf_decoder_layer.mlp.gate_proj.bias.data          # [4H]
        else:
            up_b = None
        
        # Note: HF uses gate_proj and up_proj separately: 
        #   gate_proj: [H, 4H], up_proj: [H, 4H], then they do `si lu(gate) * up`
        #   So “fused upweight” for activation is slightly different. 
        #   We replicate their logic: split “gate_proj” and “up_proj” separately in TP.

        # up2_w  = hf_decoder_layer.mlp.up_proj.weight.data            # [H, 4H]
        # up2_b  = hf_decoder_layer.mlp.up_proj.bias.data              # [4H]
        up2_w = hf_decoder_layer.mlp.up_proj.weight.data              # [H, 4H]
        if hf_decoder_layer.mlp.up_proj.bias is not None:
            up2_b = hf_decoder_layer.mlp.up_proj.bias.data            # [4H]
        else:
            up2_b = None
        
        # But in TP we want to treat gate_proj and up_proj as two separate row-parallel layers.
        # We will slice them individually. That is fine.

        # down_w = hf_decoder_layer.mlp.down_proj.weight.data         # [4H, H]
        # down_b = hf_decoder_layer.mlp.down_proj.bias.data           # [H]
        down_w = hf_decoder_layer.mlp.down_proj.weight.data          # [4H, H]
        if hf_decoder_layer.mlp.down_proj.bias is not None:
            down_b = hf_decoder_layer.mlp.down_proj.bias.data        # [H]
        else:
            down_b = None

        # Build submodules:

        # 1) Attention
        self.attn = TensorParallelLlamaAttention(
            llama_config=llama_config,
            layer_idx=layer_idx,
            world_size=world_size,
            group=group,
            pretrained_qkv_weight=fused_qkv_w,   # [H, 3H]
            pretrained_qkv_bias=fused_qkv_b,     # [3H]
            pretrained_o_weight=o_w,             # [H, H]
            pretrained_o_bias=o_b                # [H]
        )

        # 2) MLP
        #    We need two row-parallel Up projections: gate_proj and up_proj, each [H, 4H]
        #    On each rank, row-slice [H, 4H/W] for gate_proj and up_proj.
        #    Then after we do intermediate = silu(gate_local) * up_local → [batch, seq, 4H/W]
        #    We will all-gather → [batch, seq, 4H], then join down_proj.
        #
        # We'll create two RowParallelLinears for gate and up, then use a ColumnParallel for down.
        self.mlp_g_proj = RowParallelLinear(
            orig_linear=hf_decoder_layer.mlp.gate_proj,  # [H, 4H]
            world_size=world_size,
            group=group
        )
        self.mlp_up_proj = RowParallelLinear(
            orig_linear=hf_decoder_layer.mlp.up_proj,    # [H, 4H]
            world_size=world_size,
            group=group
        )
        self.mlp_down_proj = ColumnParallelLinear(
            orig_linear=hf_decoder_layer.mlp.down_proj,  # [4H, H]
            world_size=world_size,
            group=group
        )
        # The biases for those linears will be copied in `RowParallelLinear` and `ColumnParallelLinear`
        # because in their __init__ we do NOT load weights; we'll handle that manually right now:

        # Manually load weights into RowParallel gate & up:
        rank = dist.get_rank(self.group)
        # gate_proj: [H, 4H] → chunk dim=1 → [H, 4H/W]
        gate_w_chunks = split_parameter(up_w, dim=1, world_size=world_size)
        self.mlp_g_proj.weight.data.copy_(gate_w_chunks[rank])
        gate_b_chunks = split_parameter(up_b, dim=0, world_size=world_size)
        self.mlp_g_proj.bias.data.copy_(gate_b_chunks[rank])

        # up_proj: [H, 4H] → chunk dim=1 → [H, 4H/W]
        up_w_chunks = split_parameter(up2_w, dim=1, world_size=world_size)
        self.mlp_up_proj.weight.data.copy_(up_w_chunks[rank])
        up_b_chunks = split_parameter(up2_b, dim=0, world_size=world_size)
        self.mlp_up_proj.bias.data.copy_(up_b_chunks[rank])

        # down_proj: [4H, H] → chunk dim=0 → [4H/W, H]
        down_w_chunks = split_parameter(down_w, dim=0, world_size=world_size)
        self.mlp_down_proj.weight.data.copy_(down_w_chunks[rank])
        # bias is full [H], shared on each rank
        self.mlp_down_proj.bias.data.copy_(down_b)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch, seq, H]
        Returns: [batch, seq, H]
        """
        # 1) After attention, the residual + dropout has been applied already by the attention block.
        #    Here, we just perform MLP on the result.

        # a) Compute gate and up locally:
        #    Each is RowParallel: input [batch, seq, H] → local linear → [batch, seq, 4H/W]
        gate_local = self.mlp_g_proj(hidden_states)   # [batch, seq, 4H/W]
        up_local   = self.mlp_up_proj(hidden_states)  # [batch, seq, 4H/W]

        # b) Activation & elementwise multiply: [batch, seq, 4H/W]
        intermediate_local = F.silu(gate_local) * up_local

        # c) All-gather intermediate shards → [batch, seq, 4H]
        intermediate_full = all_gather_tensor(intermediate_local, self.group)

        # d) Column-parallel down: [batch, seq, H]
        mlp_output = self.mlp_down_proj(intermediate_full)  # automatically all-reduces → [batch, seq, H]

        return mlp_output


# --------------------------------------------------------------------------------
# 6) TensorParallelLlamaModel
#
#    Replaces the HuggingFace LlamaModel. We build:
#      - embed_tokens (possibly un-sharded)
#      - embed_positions (not used in Llama; uses rotary)
#      - a stack of N TensorParallelLlamaDecoderLayer
#      - final layernorm + (if LM) final head etc.
#
#    For classification, we will just use the hidden state at position 0.
# --------------------------------------------------------------------------------

from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM

class TensorParallelLlamaModel(LlamaPreTrainedModel):
    """
    Wraps a full LlamaModel in Tensor‐Parallel layers.
    We call the HF LlamaModel from_pretrained, extract its weights,
    then we build our own TP layers and copy in the correct shards.
    """
    def __init__(self,
                 config: LlamaConfig,
                 world_size: int,
                 group: torch.distributed.ProcessGroup,
                 pretrained_model: LlamaModel):
        """
        config: LlamaConfig
        world_size, group: for distributed
        pretrained_model: a loaded HuggingFace LlamaModel (CPU or GPU)
        """
        super().__init__(config)
        self.world_size = world_size
        self.group = group

        # 1) Embedding (vocab and position)
        #    We choose to keep embedding un-sharded. If you need to shard it, you can apply a vocab-split.
        #    For now, each rank keeps the full embedding.
        self.embed_tokens = nn.Embedding(
            pretrained_model.embed_tokens.num_embeddings,
            pretrained_model.embed_tokens.embedding_dim,
            _weight=pretrained_model.embed_tokens.weight.data.clone()
        )

        # 2) Final output layernorm
        self.final_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm.weight.data.copy_(pretrained_model.norm.weight.data)

        # 3) Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx, hf_layer in enumerate(pretrained_model.layers):
            # For each HF LlamaDecoderLayer, we extract it and pass it in to our TP version.
            tp_layer = TensorParallelLlamaDecoderLayer(
                llama_config=config,
                layer_idx=layer_idx,
                world_size=world_size,
                group=group,
                hf_decoder_layer=hf_layer
            )
            self.layers.append(tp_layer)

        # 4) Cache hidden size
        self.hidden_size = config.hidden_size

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                **kwargs):
        """
        input_ids: [batch, seq]
        attention_mask: [batch, seq] with 1=keep, 0=mask
        position_ids:   [batch, seq] or None

        Returns:
          last_hidden_state: [batch, seq, H]
          hidden_states (optional)
          attentions (optional)
        """
        # 1) Embedding lookup: [batch, seq, H]
        hidden_states = self.embed_tokens(input_ids)

        # 2) Build causal mask for attention if none provided
        if attention_mask is None:
            seq_len = input_ids.size(1)
            attention_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq, seq]
        else:
            # HuggingFace expects mask shape [batch, 1, 1, seq]
            attention_mask = attention_mask.view(input_ids.size(0), 1, 1, input_ids.size(1))

        # 3) Pass through each TP layer
        all_hidden_states = [] if kwargs.get("output_hidden_states", False) else None
        all_attentions    = [] if kwargs.get("output_attentions", False) else None

        for layer_idx, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            h, attn = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            hidden_states = h
            if all_attentions is not None:
                all_attentions.append(attn)

        # 4) Final LayerNorm
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states,)
        if all_hidden_states is not None:
            outputs = outputs + (all_hidden_states,)
        if all_attentions is not None:
            outputs = outputs + (all_attentions,)

        return outputs  # (last_hidden_state, [hidden_states], [attentions])
