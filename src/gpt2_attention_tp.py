import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, Conv1D
from tensor_parallel_with_lossy import LinearShardedOutputsLossy, LinearShardedInputsLossy, LossyAllReduceFwdIdentityBwd


class TensorParallelGPT2Attention(GPT2Attention):
    """
    Tensor Parallel GPT2 Attention with Lossy Network Simulation
    """
    def __init__(self, original_attention, group, lossy_network):
        super().__init__(original_attention.config, original_attention.layer_idx)
        
        self.group = group
        self.lossy_network = lossy_network
        self.config = original_attention.config
        self.layer_idx = original_attention.layer_idx
        
        max_positions = original_attention.config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = original_attention.embed_dim
        self.num_heads = original_attention.num_heads
        self.head_dim = original_attention.head_dim
        self.split_size = original_attention.split_size
        self.scale_attn_weights = original_attention.scale_attn_weights
        self.is_cross_attention = original_attention.is_cross_attention
        
        # Replace c_attn with separate Q, K, V projections that are column parallel
        self.c_attn_original = original_attention.c_attn
        
        # Extract the original combined QKV weights
        if isinstance(self.c_attn_original, Conv1D):
            # Conv1D stores weight as [in_features, out_features] 
            in_features, total_out_features = self.c_attn_original.weight.shape
            qkv_weight = self.c_attn_original.weight.data.transpose(0, 1)  # [3*embed_dim, embed_dim]
            qkv_bias = self.c_attn_original.bias.data if self.c_attn_original.bias is not None else None
        else:
            # Regular Linear layer
            total_out_features, in_features = self.c_attn_original.weight.shape
            qkv_weight = self.c_attn_original.weight.data  # [3*embed_dim, embed_dim]
            qkv_bias = self.c_attn_original.bias.data if self.c_attn_original.bias is not None else None
        
        assert total_out_features == 3 * self.embed_dim, f"Expected {3 * self.embed_dim}, got {total_out_features}"
        
        # Split QKV weights into separate Q, K, V
        q_weight = qkv_weight[:self.embed_dim]  # [embed_dim, embed_dim]
        k_weight = qkv_weight[self.embed_dim:2*self.embed_dim]  # [embed_dim, embed_dim]
        v_weight = qkv_weight[2*self.embed_dim:]  # [embed_dim, embed_dim]
        
        if qkv_bias is not None:
            q_bias = qkv_bias[:self.embed_dim]
            k_bias = qkv_bias[self.embed_dim:2*self.embed_dim] 
            v_bias = qkv_bias[2*self.embed_dim:]
        else:
            q_bias = k_bias = v_bias = None
        
        # Create tensor parallel Q, K, V projections (column parallel)
        # Each rank handles a subset of attention heads
        world_size = group.size()
        assert self.num_heads % world_size == 0, f"num_heads ({self.num_heads}) must be divisible by world_size ({world_size})"
        
        local_num_heads = self.num_heads // world_size
        local_embed_dim = self.embed_dim // world_size  # This matches what LinearShardedOutputsLossy calculates
        
        self.q_proj = LinearShardedOutputsLossy(
            in_features, self.embed_dim, group, lossy_network,
            device=self.c_attn_original.weight.device, 
            dtype=self.c_attn_original.weight.dtype
        )
        self.k_proj = LinearShardedOutputsLossy(
            in_features, self.embed_dim, group, lossy_network,
            device=self.c_attn_original.weight.device,
            dtype=self.c_attn_original.weight.dtype
        )
        self.v_proj = LinearShardedOutputsLossy(
            in_features, self.embed_dim, group, lossy_network,
            device=self.c_attn_original.weight.device,
            dtype=self.c_attn_original.weight.dtype
        )
        
        # Initialize weights with the correct shards (head-wise sharding)
        rank = dist.get_rank(group)
        head_start = rank * local_num_heads
        head_end = head_start + local_num_heads
        
        # Reshape weights to be head-wise: [num_heads, head_dim, embed_dim]
        q_weight_heads = q_weight.view(self.num_heads, self.head_dim, in_features)
        k_weight_heads = k_weight.view(self.num_heads, self.head_dim, in_features)
        v_weight_heads = v_weight.view(self.num_heads, self.head_dim, in_features)
        
        # Take the heads for this rank and flatten back
        q_local = q_weight_heads[head_start:head_end].view(local_embed_dim, in_features)
        k_local = k_weight_heads[head_start:head_end].view(local_embed_dim, in_features)
        v_local = v_weight_heads[head_start:head_end].view(local_embed_dim, in_features)
        
        self.q_proj.weight.data.copy_(q_local)
        self.k_proj.weight.data.copy_(k_local)
        self.v_proj.weight.data.copy_(v_local)
        
        if q_bias is not None:
            q_bias_heads = q_bias.view(self.num_heads, self.head_dim)
            k_bias_heads = k_bias.view(self.num_heads, self.head_dim)
            v_bias_heads = v_bias.view(self.num_heads, self.head_dim)
            
            self.q_proj.bias.data.copy_(q_bias_heads[head_start:head_end].view(local_embed_dim))
            self.k_proj.bias.data.copy_(k_bias_heads[head_start:head_end].view(local_embed_dim))
            self.v_proj.bias.data.copy_(v_bias_heads[head_start:head_end].view(local_embed_dim))
        
        # Store local dimensions
        self.local_num_heads = local_num_heads
        self.local_embed_dim = local_embed_dim
        
        # Output projection as row parallel - takes local attention output and produces full embed_dim
        self.c_proj = nn.Linear(
            local_embed_dim, self.embed_dim, bias=(original_attention.c_proj.bias is not None),
            device=original_attention.c_proj.weight.device,
            dtype=original_attention.c_proj.weight.dtype
        )
        
        # Initialize c_proj weights - need to take the columns corresponding to our heads
        if isinstance(original_attention.c_proj, Conv1D):
            # Conv1D weight needs transposing: [embed_dim, embed_dim] -> [embed_dim, embed_dim]
            c_proj_weight = original_attention.c_proj.weight.data.transpose(0, 1)  # [embed_dim, embed_dim]
        else:
            c_proj_weight = original_attention.c_proj.weight.data
        
        # Reshape to head-wise: [embed_dim, num_heads, head_dim]
        c_proj_heads = c_proj_weight.view(self.embed_dim, self.num_heads, self.head_dim)
        # Take columns for our heads and flatten: [embed_dim, local_embed_dim]
        c_proj_local = c_proj_heads[:, head_start:head_end, :].view(self.embed_dim, local_embed_dim)
        
        self.c_proj.weight.data.copy_(c_proj_local)  # nn.Linear expects [out_features, in_features] = [embed_dim, local_embed_dim]
        if original_attention.c_proj.bias is not None:
            self.c_proj.bias.data.copy_(original_attention.c_proj.bias.data)
        
        self.attn_dropout = original_attention.attn_dropout
        self.resid_dropout = original_attention.resid_dropout
        
        # Remove the original c_attn to avoid confusion
        delattr(self, 'c_attn_original')
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.config.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        past_key_value=None,  # Added to match GPT2 interface
        **kwargs,  # Catch any other unexpected arguments
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn`, `c_attn`, `c_proj` have to be defined. "
                    f"Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # Apply tensor parallel Q, K, V projections
            # These produce local head outputs: [batch, seq, local_embed_dim]
            query = self.q_proj(hidden_states)
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

        # Handle both layer_past and past_key_value interfaces
        layer_past = layer_past if layer_past is not None else past_key_value

        # Split heads for local computation
        query = self._split_heads(query, self.local_num_heads, self.head_dim)
        key = self._split_heads(key, self.local_num_heads, self.head_dim)
        value = self._split_heads(value, self.local_num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Merge heads back - this gives us [batch, seq, local_embed_dim]
        attn_output = self._merge_heads(attn_output, self.local_num_heads, self.head_dim)
        
        # Apply output projection locally
        attn_output = self.c_proj(attn_output)
        
        # Apply lossy all-reduce to combine outputs from all ranks
        attn_output = LossyAllReduceFwdIdentityBwd.apply(attn_output, self.group, self.lossy_network)
        
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


def replace_gpt2_attention_with_tp_lossy(module, group, lossy_network):
    """
    Replace GPT2 attention layers with tensor parallel versions
    """
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    
    for name, child in list(module.named_children()):
        if isinstance(child, GPT2Attention):
            # Replace with tensor parallel version
            new_attention = TensorParallelGPT2Attention(child, group, lossy_network)
            setattr(module, name, new_attention)
            print(f"Replaced {name} with TensorParallelGPT2Attention")
        else:
            # Recursively apply to child modules
            replace_gpt2_attention_with_tp_lossy(child, group, lossy_network)
