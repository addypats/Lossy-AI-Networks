
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


# tp_Llama_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)


# --------------------------------------------------------------------------------
# Utility functions for weight sharding and collective ops
# --------------------------------------------------------------------------------

def split_parameter(tensor: torch.Tensor, dim: int, world_size: int):
    """
    Splits `tensor` into `world_size` equal chunks along `dim`, returns a list of chunks.
    """
    assert tensor.size(dim) % world_size == 0, \
        f"Dimension {dim}={tensor.size(dim)} not divisible by world_size={world_size}"
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

        # Allocate a local weight shard on orig_linear.weight.device
        self.weight = nn.Parameter(
            torch.empty(self.local_out, in_features, device=orig_linear.weight.device)
        )
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(
                torch.empty(self.local_out, device=orig_linear.weight.device)
            )
        else:
            self.bias = None

        # We do NOT initialize here; pretrained copying happens externally
        self._orig_out = out_features
        self._orig_in = in_features

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
# --------------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, world_size: int, group: torch.distributed.ProcessGroup):
        """
        Creates a column-parallel wrapper around a pretrained nn.Linear.
        Original Linear: weight shape [out_f, in_f]
        We assert in_f % world_size == 0.
        Each rank gets a local weight [out_f, in_f/world_size].

        Forward:
          - We assume x is full [batch, in_f].
          - Slice columns: x[:, rank*local_in:(rank+1)*local_in] → [batch, local_in].
          - Multiply with local_weight.T ([local_in, out_f]) → [batch, out_f].
          - All-reduce sum partial outputs → final [batch, out_f].
        """
        super().__init__()
        self.world_size = world_size
        self.group = group

        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        assert in_features % world_size == 0, \
            f"ColumnParallel: in_features ({in_features}) not divisible by world_size ({world_size})"
        self.local_in = in_features // world_size

        # Each rank’s local weight slice: [out_features, local_in]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in, device=orig_linear.weight.device)
        )
        if orig_linear.bias is not None:
            # Bias is shared, so we replicate full-size bias on each rank
            self.bias = nn.Parameter(torch.empty(out_features, device=orig_linear.weight.device))
        else:
            self.bias = None

        self._orig_out = out_features
        self._orig_in = in_features

    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, orig_in_features]
        We assume the caller has already provided the *full* input x.
        Each rank will slice its local columns: x[:, rank*local_in:(rank+1)*local_in]
        Then compute local matmul and all-reduce to sum partials.

        Returns: [batch_size, out_features]
        """
        rank = dist.get_rank(self.group)
        start = rank * self.local_in
        end = start + self.local_in
        x_local = x[:, start:end]  # [batch, local_in]

        # F.linear(input, weight, bias) does input @ weight.T + bias
        # Here weight.T is [local_in, out_features]
        local_output = F.linear(x_local, self.weight.t(), None)  # [batch, out_features]

        # All-reduce to sum partial outputs
        output_full = all_reduce_tensor(local_output, self.group)  # [batch, out_features]
        if self.bias is not None:
            output_full = output_full + self.bias
        return output_full


# --------------------------------------------------------------------------------
# 3) TensorParallelLlamaAttention
# --------------------------------------------------------------------------------

from transformers.models.llama.modeling_llama import LlamaAttention

class TensorParallelLlamaAttention(LlamaAttention):
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
        llama_config: HF LlamaConfig
        pretrained_qkv_weight: full [H, 3H] weight from pretrained qkv_proj (or fused)
        pretrained_qkv_bias:   full [3H] bias or None
        pretrained_o_weight:   full [H, H] weight from pretrained o_proj
        pretrained_o_bias:     full [H] bias or None
        """
        super().__init__(llama_config, layer_idx)

        self.world_size = world_size
        self.group = group
        self.hidden_size = llama_config.hidden_size
        self.num_heads = llama_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        assert self.hidden_size % world_size == 0, "hidden_size must be divisible by world_size"
        # Ensure head_dim is also divisible by world_size
        assert (self.head_dim % world_size) == 0, "head_dim must be divisible by world_size"
        self.local_hidden = self.hidden_size // world_size  # H/W

        # Initialize rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(
            base=llama_config.rotary_embedding_base,
            head_dim=self.head_dim,
            freqs_for="vertical"
        )

        # LayerNorm before attention
        self.input_layernorm = LlamaRMSNorm(self.hidden_size, eps=llama_config.rms_norm_eps)

        # 1) Row-parallel QKV projection: splits fused [H, 3H] → [H, 3H/W]
        self.qkv_weight = nn.Parameter(
            torch.empty(self.hidden_size, (3 * self.hidden_size) // world_size,
                         device=pretrained_qkv_weight.device)
        )
        if pretrained_qkv_bias is not None:
            self.qkv_bias = nn.Parameter(
                torch.empty((3 * self.hidden_size) // world_size,
                             device=pretrained_qkv_bias.device)
            )
        else:
            self.qkv_bias = None

        # 2) Column-parallel O projection: each rank holds [H/W, H]
        self.o_weight = nn.Parameter(
            torch.empty(self.local_hidden, self.hidden_size,
                         device=pretrained_o_weight.device)
        )
        if pretrained_o_bias is not None:
            # Bias is full [H], replicated on each rank
            self.o_bias = nn.Parameter(torch.empty(self.hidden_size, device=pretrained_o_bias.device))
        else:
            self.o_bias = None

        # Dropout layers
        self.attn_dropout = nn.Dropout(llama_config.attention_dropout_prob)
        self.resid_dropout = nn.Dropout(llama_config.hidden_dropout_prob)

        # -----------------------------------------------------------
        # Load pretrained shards into our parameters
        rank = dist.get_rank(self.group)

        # a) Split full qkv weight [H, 3H] → world_size chunks along dim=1
        qkv_chunks = split_parameter(pretrained_qkv_weight, dim=1, world_size=world_size)
        self.qkv_weight.data.copy_(qkv_chunks[rank])  # [H, 3H/W]
        if pretrained_qkv_bias is not None:
            bias_chunks = split_parameter(pretrained_qkv_bias, dim=0, world_size=world_size)
            self.qkv_bias.data.copy_(bias_chunks[rank])

        # b) Split full o_proj weight [H, H] → world_size row-wise chunks [H/W, H]
        o_weight_chunks = split_parameter(pretrained_o_weight, dim=0, world_size=world_size)
        self.o_weight.data.copy_(o_weight_chunks[rank])  # [H/W, H]
        if pretrained_o_bias is not None:
            # Bias is full [H], replicated
            self.o_bias.data.copy_(pretrained_o_bias)

        self._orig_qkv_out = 3 * self.hidden_size
        self._orig_o_out = self.hidden_size

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                **kwargs):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, 1, 1, seq_len] (0=mask, 1=keep)
        position_ids:   [batch_size, seq_len] if used with rotary

        Returns:
          new_hidden_states: [batch_size, seq_len, hidden_size]
          attn_probs:       [batch_size, num_heads, seq_len, seq_len]
        """
        # 1) Pre-LN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)  # [b, seq, H]

        # 2) Fused QKV projection (row-parallel)
        #    local_qkv: [b, seq, 3*(H/W)]
        local_qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)

        # 3) Chunk into q_local, k_local, v_local each [b, seq, H/W]
        q_local, k_local, v_local = torch.chunk(local_qkv, 3, dim=-1)

        # 4) Reshape to [b, num_heads, seq, head_dim/W]
        new_shape = (
            q_local.size(0),
            self.num_heads,
            q_local.size(1),
            self.head_dim // self.world_size,
        )
        q_local = q_local.view(q_local.size(0), q_local.size(1), self.num_heads, self.head_dim // self.world_size) \
                       .transpose(1, 2)  # [b, num_heads, seq, head_dim/W]
        k_local = k_local.view(k_local.size(0), k_local.size(1), self.num_heads, self.head_dim // self.world_size) \
                       .transpose(1, 2)
        v_local = v_local.view(v_local.size(0), v_local.size(1), self.num_heads, self.head_dim // self.world_size) \
                       .transpose(1, 2)

        # 5) Apply rotary embeddings if provided
        if position_ids is not None:
            q_local, k_local = self.rotary_emb(q_local, k_local, position_ids)

        # 6) All-gather K_local and V_local → [b, num_heads, seq, head_dim]
        k_gathered = all_gather_tensor(k_local, self.group)
        v_gathered = all_gather_tensor(v_local, self.group)

        # 7) Compute attention scores: [b, num_heads, seq, seq]
        attn_scores = torch.matmul(
            q_local,
            k_gathered.transpose(-1, -2)
        ) / (self.head_dim ** 0.5)

        # 8) Apply mask
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 9) Compute attention output: [b, num_heads, seq, head_dim]
        attn_output_local = torch.matmul(attn_probs, v_gathered)

        # 10) Merge heads → [b, seq, H/W]
        attn_output_local = (
            attn_output_local.transpose(1, 2).contiguous()
            .view(attn_output_local.size(0), attn_output_local.size(2), self.local_hidden)
        )  # [b, seq, H/W]

        # 11) Row-parallel O projection: gather full [b, seq, H]
        attn_output_full = all_gather_tensor(attn_output_local, self.group)  # [b, seq, H]

        # 12) Local O matmul: [b, seq, H/W] = [b, seq, H] @ weight.T where weight=[H/W, H]
        local_o_out = F.linear(attn_output_full, self.o_weight, None)  # [b, seq, H/W]

        # 13) All-gather partial O outputs → [b, seq, H]
        o_out_full = all_gather_tensor(local_o_out, self.group)  # [b, seq, H]
        if self.o_bias is not None:
            o_out_full = o_out_full + self.o_bias

        # 14) Residual + dropout
        o_out_full = self.resid_dropout(o_out_full)
        o_out_full = residual + o_out_full

        return o_out_full, attn_probs


# --------------------------------------------------------------------------------
# 4) TensorParallelLlamaMLP
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
        llama_config: HF LlamaConfig
        pretrained_up_weight:   full [H, 4H]
        pretrained_up_bias:     full [4H] or None
        pretrained_down_weight: full [H, 4H]
        pretrained_down_bias:   full [H] or None
        """
        super().__init__()
        self.world_size = world_size
        self.group = group
        self.hidden_size = llama_config.hidden_size
        self.intermediate_size = llama_config.intermediate_size  # = 4H
        assert self.intermediate_size == 4 * self.hidden_size, "intermediate_size must equal 4 * hidden_size"

        # LayerNorm after attention
        self.post_attn_layernorm = LlamaRMSNorm(self.hidden_size, eps=llama_config.rms_norm_eps)

        # Row-parallel “up”:
        #   Each rank’s up_weight shard: [H, 4H/W]
        self.up_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size // world_size,
                        device=pretrained_up_weight.device)
        )
        if pretrained_up_bias is not None:
            self.up_bias = nn.Parameter(
                torch.empty(self.intermediate_size // world_size, device=pretrained_up_bias.device)
            )
        else:
            self.up_bias = None

        # Column-parallel “down”:
        #   Each rank’s down_weight shard: [H, 4H/W]
        self.down_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size // world_size,
                        device=pretrained_down_weight.device)
        )
        if pretrained_down_bias is not None:
            # This full bias [H] is replicated on each rank
            self.down_bias = nn.Parameter(torch.empty(self.hidden_size, device=pretrained_down_bias.device))
        else:
            self.down_bias = None

        # Dropout
        self.resid_dropout = nn.Dropout(llama_config.hidden_dropout_prob)

        # ──────────── Load Pretrained Shards ────────────
        rank = dist.get_rank(self.group)

        # 1) Split up_weight [H, 4H] → chunks along dim=1 → each chunk = [H, 4H/W]
        up_weight_chunks = split_parameter(pretrained_up_weight, dim=1, world_size=world_size)
        self.up_weight.data.copy_(up_weight_chunks[rank])
        if pretrained_up_bias is not None:
            up_bias_chunks = split_parameter(pretrained_up_bias, dim=0, world_size=world_size)
            self.up_bias.data.copy_(up_bias_chunks[rank])

        # 2) Split down_weight [H, 4H] → chunks along dim=1 → each chunk = [H, 4H/W]
        full_down_weight = pretrained_down_weight  # shape [H, 4H]
        chunk_size = self.intermediate_size // world_size  # = 4H/W
        start = rank * chunk_size
        end = (rank + 1) * chunk_size
        # Each local down_weight: [H, 4H/W]
        self.down_weight.data.copy_(full_down_weight[:, start:end])
        if pretrained_down_bias is not None:
            # Bias is full [H], replicated
            self.down_bias.data.copy_(pretrained_down_bias)

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch, seq, H]
        Returns: [batch, seq, H]
        """
        # 1) Pre‐MLP LayerNorm
        residual = hidden_states
        hidden_states = self.post_attn_layernorm(hidden_states)  # [b, seq, H]

        # 2) Row-parallel “up”: project H → 4H/W on each rank
        #    local_intermediate: [b, seq, 4H/W]
        local_intermediate = F.linear(hidden_states, self.up_weight, self.up_bias)

        # 3) Activation
        local_intermediate = F.silu(local_intermediate)  # [b, seq, 4H/W]

        # 4) All-gather local_intermediate → [batch, seq, 4H]
        intermediate_full = all_gather_tensor(local_intermediate, self.group)  # [b, seq, 4H]

        # 5) Column-parallel “down”: each rank’s down_weight: [H, 4H/W]
        #    We must slice intermediate_full along last dim:
        rank = dist.get_rank(self.group)
        chunk_size = self.intermediate_size // self.world_size  # = 4H/W
        start = rank * chunk_size
        end = (rank + 1) * chunk_size
        intermediate_local_slice = intermediate_full[:, :, start:end]  # [b, seq, 4H/W]

        # 6) Local down matmul → [batch, seq, H]:
        #    We have down_weight of shape [H, 4H/W], so we multiply:
        #    intermediate_local_slice [b, seq, 4H/W] @ down_weight.T [4H/W, H] → [b, seq, H]
        local_mlp_output = F.linear(intermediate_local_slice, self.down_weight.T, None)  # [b, seq, H]

        # 7) All-reduce across ranks → final [batch, seq, H]
        mlp_output_full = all_reduce_tensor(local_mlp_output, self.group)  # [b, seq, H]
        if self.down_bias is not None:
            mlp_output_full = mlp_output_full + self.down_bias

        # 8) Residual + dropout
        mlp_output_full = self.resid_dropout(mlp_output_full)
        mlp_output_full = residual + mlp_output_full

        return mlp_output_full


# --------------------------------------------------------------------------------
# 5) TensorParallelLlamaDecoderLayer
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
        hf_decoder_layer: the HF LlamaDecoderLayer instance to extract pretrained weights
        """
        super().__init__()
        self.world_size = world_size
        self.group = group
        self.config = llama_config

        H = llama_config.hidden_size

        # ─── Build fused QKV weight & bias ───
        if hasattr(hf_decoder_layer.self_attn, "qkv_proj"):
            raw_qkv = hf_decoder_layer.self_attn.qkv_proj.weight.data  # [H, 3H]
            if raw_qkv.shape != (H, 3 * H):
                raise RuntimeError(
                    f"Found fused qkv_proj, but its weight is {raw_qkv.shape}, expected ({H}, {3*H})"
                )
            fused_qkv_w = raw_qkv.clone()
            if (hf_decoder_layer.self_attn.qkv_proj.bias is not None
                    and hf_decoder_layer.self_attn.qkv_proj.bias.data.shape[0] == 3 * H):
                fused_qkv_b = hf_decoder_layer.self_attn.qkv_proj.bias.data.clone()
            else:
                fused_qkv_b = None

        elif (hasattr(hf_decoder_layer.self_attn, "q_proj")
              and hasattr(hf_decoder_layer.self_attn, "k_proj")
              and hasattr(hf_decoder_layer.self_attn, "v_proj")):
            # Build fused QKV from separate q_proj/k_proj/v_proj
            raw_q_w = hf_decoder_layer.self_attn.q_proj.weight.data  # [H, H]
            raw_k_w = hf_decoder_layer.self_attn.k_proj.weight.data
            raw_v_w = hf_decoder_layer.self_attn.v_proj.weight.data

            # Ensure each is [H, H] by transposing if needed
            if raw_q_w.shape != (H, H):
                raise RuntimeError(f"Unexpected q_proj shape {raw_q_w.shape}, expected [H, H].")
            q_w = raw_q_w

            # K
            if raw_k_w.shape == (H, H):
                k_w = raw_k_w
            elif raw_k_w.shape[1] == H:
                k_w = raw_k_w.transpose(0, 1)
            else:
                raise RuntimeError(f"k_proj has shape {raw_k_w.shape}, which is neither [H, Dk] nor [Dk, H].")

            # V
            if raw_v_w.shape == (H, H):
                v_w = raw_v_w
            elif raw_v_w.shape[1] == H:
                v_w = raw_v_w.transpose(0, 1)
            else:
                raise RuntimeError(f"v_proj has shape {raw_v_w.shape}, which is neither [H, Dv] nor [Dv, H].")

            # Now q_w, k_w, v_w are each [H, H]
            fused_qkv_w = torch.cat([q_w, k_w, v_w], dim=1)  # [H, 3H]

            # Handle biases
            raw_q_b = hf_decoder_layer.self_attn.q_proj.bias.data \
                if hf_decoder_layer.self_attn.q_proj.bias is not None else None
            raw_k_b = hf_decoder_layer.self_attn.k_proj.bias.data \
                if hf_decoder_layer.self_attn.k_proj.bias is not None else None
            raw_v_b = hf_decoder_layer.self_attn.v_proj.bias.data \
                if hf_decoder_layer.self_attn.v_proj.bias is not None else None

            if raw_q_b is not None and raw_k_b is not None and raw_v_b is not None:
                # All must be [H]
                k_b = raw_k_b if raw_k_b.shape == (H,) else raw_k_b.repeat_interleave(H // raw_k_b.shape[0])
                v_b = raw_v_b if raw_v_b.shape == (H,) else raw_v_b.repeat_interleave(H // raw_v_b.shape[0])
                fused_qkv_b = torch.cat([raw_q_b, k_b, v_b], dim=0)  # [3H]
            else:
                fused_qkv_b = None

        else:
            all_names = [n for n, _ in hf_decoder_layer.self_attn.named_parameters()]
            raise RuntimeError(
                "Cannot find a valid QKV in this LlamaAttention. "
                f"self_attn parameters = {all_names}"
            )

        # 2) O projection weight & bias
        o_w = hf_decoder_layer.self_attn.o_proj.weight.data.clone()  # [H, H]
        o_b = hf_decoder_layer.self_attn.o_proj.bias.data.clone() \
            if hf_decoder_layer.self_attn.o_proj.bias is not None else None

        # 3) MLP projections:
        #    gate_proj: [H, 4H]
        #    up_proj:   [H, 4H]
        #    down_proj: [4H, H]
        gate_w = hf_decoder_layer.mlp.gate_proj.weight.data.clone()  # [H, 4H]
        gate_b = hf_decoder_layer.mlp.gate_proj.bias.data.clone() \
            if hf_decoder_layer.mlp.gate_proj.bias is not None else None

        up2_w = hf_decoder_layer.mlp.up_proj.weight.data.clone()  # [H, 4H]
        up2_b = hf_decoder_layer.mlp.up_proj.bias.data.clone() \
            if hf_decoder_layer.mlp.up_proj.bias is not None else None

        down_w = hf_decoder_layer.mlp.down_proj.weight.data.clone()  # [4H, H]
        down_b = hf_decoder_layer.mlp.down_proj.bias.data.clone() \
            if hf_decoder_layer.mlp.down_proj.bias is not None else None

        # Build submodules:

        # 1) Attention
        self.attn = TensorParallelLlamaAttention(
            llama_config=llama_config,
            layer_idx=layer_idx,
            world_size=world_size,
            group=group,
            pretrained_qkv_weight=fused_qkv_w,    # [H, 3H]
            pretrained_qkv_bias=fused_qkv_b,      # [3H]
            pretrained_o_weight=o_w,              # [H, H]
            pretrained_o_bias=o_b                 # [H]
        )

        # 2) MLP
        #    Row-parallel gate_proj and up_proj (each [H, 4H]) → shard along dim=1
        #    Column-parallel down_proj: full [4H, H] → we'll slice along columns
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

        # Manually load shards into gate_proj & up_proj
        rank = dist.get_rank(group)
        # gate_proj: [H, 4H] → dim=1 → [H, 4H/W]
        gate_w_chunks = split_parameter(gate_w, dim=1, world_size=world_size)
        self.mlp_g_proj.weight.data.copy_(gate_w_chunks[rank])
        if gate_b is not None:
            gate_b_chunks = split_parameter(gate_b, dim=0, world_size=world_size)
            self.mlp_g_proj.bias.data.copy_(gate_b_chunks[rank])

        # up_proj: [H, 4H] → dim=1 → [H, 4H/W]
        up2_w_chunks = split_parameter(up2_w, dim=1, world_size=world_size)
        self.mlp_up_proj.weight.data.copy_(up2_w_chunks[rank])
        if up2_b is not None:
            up2_b_chunks = split_parameter(up2_b, dim=0, world_size=world_size)
            self.mlp_up_proj.bias.data.copy_(up2_b_chunks[rank])

        # down_proj: ColumnParallelLinear will handle slicing from full [4H, H]
        # But HF’s down_proj is [4H, H], while ColumnParallelLinear expects orig_linear.in_features = H
        # Actually, ColumnParallelLinear expects weight [out_f, in_f], i.e. out_f=4H, in_f=H.
        # That matches if we treat each linear as weight [4H, H].
        # In ColumnParallelLinear, each rank’s local weight is [4H, H/W]? No: it expects
        # in_f % world_size == 0 and weight shape = [out_f, in_f/world_size]. 
        # But here, in_f=H. We need H % W == 0 → that holds if H divisible by W. 
        # So each rank gets [4H, H/W]. F.linear does input_local [b, seq, H/W] @ weight.T [H/W, 4H] 
        # → [b, seq, 4H]. Then all-reduce→ [b, seq, 4H]. That fits the intended flow.
        # Finally we need to project [b, seq, 4H] → [b, seq, H], but ColumnParallelLinear will do that if
        # orig_linear = Linear(in_features=H, out_features=4H)? Actually our orig_linear has in_features=H, out_features=4H.
        # That is reversed. Instead, treat down_proj as Linear(4H→H). Then weight shape = [H, 4H]. 
        # ColumnParallelLinear expects orig_linear.in_features = 4H, orig_linear.out_features = H. So the orig_linear is correct.
        # ColumnParallelLinear splits 4H (in_features) by W. So each rank’s weight slice is [H, 4H/W].
        # In forward: each rank slices input_full along dim=-1 → chunk [b, seq, 4H/W], multiplies by weight.T → [b, seq, H], all-reduce sum → full [b, seq, H].

    def forward(self, hidden_states: torch.Tensor):
        """
        hidden_states: [batch, seq, H]
        Returns: [batch, seq, H]
        """
        # 1) Attention + Residual (inside attention returns residual added)
        attn_output, _ = self.attn(
            hidden_states,
            attention_mask=None,  # Expect attention_mask to be provided externally if needed
            position_ids=None
        )
        # 2) MLP on attn_output
        # a) Row-parallel gate and up
        gate_local = self.mlp_g_proj(attn_output)   # [b, seq, 4H/W]
        up_local = self.mlp_up_proj(attn_output)    # [b, seq, 4H/W]

        # b) Activation + elementwise multiply → [b, seq, 4H/W]
        intermediate_local = F.silu(gate_local) * up_local

        # c) All-gather → [b, seq, 4H]
        intermediate_full = all_gather_tensor(intermediate_local, self.group)

        # d) Column-parallel down → [b, seq, H]
        mlp_output = self.mlp_down_proj(intermediate_full)

        return mlp_output


# --------------------------------------------------------------------------------
# 6) TensorParallelLlamaModel
# --------------------------------------------------------------------------------

from transformers.models.llama.modeling_llama import LlamaModel

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
        pretrained_model: a loaded HF LlamaModel (CPU or GPU)
        """
        super().__init__(config)
        self.world_size = world_size
        self.group = group

        # 1) Embedding (keep un-sharded)
        self.embed_tokens = nn.Embedding(
            pretrained_model.embed_tokens.num_embeddings,
            pretrained_model.embed_tokens.embedding_dim,
            _weight=pretrained_model.embed_tokens.weight.data.clone()
        )

        # 2) Final layernorm
        self.final_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm.weight.data.copy_(pretrained_model.norm.weight.data)

        # 3) Transformer layers
        self.layers = nn.ModuleList()
        for layer_idx, hf_layer in enumerate(pretrained_model.layers):
            tp_layer = TensorParallelLlamaDecoderLayer(
                llama_config=config,
                layer_idx=layer_idx,
                world_size=world_size,
                group=group,
                hf_decoder_layer=hf_layer
            )
            self.layers.append(tp_layer)

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
          (optionally hidden_states, attentions)
        """
        # 1) Embedding lookup → [batch, seq, H]
        hidden_states = self.embed_tokens(input_ids)

        # 2) Build causal mask if none provided
        if attention_mask is None:
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device))
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq, seq]
        else:
            attention_mask = attention_mask.view(input_ids.size(0), 1, 1, input_ids.size(1))

        # 3) Pass through each TP layer
        all_hidden_states = [] if kwargs.get("output_hidden_states", False) else None
        all_attentions = [] if kwargs.get("output_attentions", False) else None

        for layer_idx, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            h = layer(hidden_states)
            hidden_states = h  # MLP returns the residual‐added output
            # Note: If you need attentions, modify TPLA to return attn_probs and collect here.

        # 4) Final LayerNorm
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states,)
        if all_hidden_states is not None:
            outputs = outputs + (all_hidden_states,)
        if all_attentions is not None:
            outputs = outputs + (all_attentions,)
        return outputs
