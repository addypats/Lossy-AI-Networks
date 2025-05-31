# import torch
# import torch.nn as nn
# from transformers.modeling_outputs import SequenceClassifierOutput

# class TPLlamaForSequenceClassification(nn.Module):
#     def __init__(self, config, world_size, group):
#         super().__init__()
#         from tp_Llama_modules import TensorParallelLlamaModel
#         self.config = config
#         self.model = TensorParallelLlamaModel(config, world_size, group)
#         self.num_labels = config.num_labels
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

#         # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_dim]
#         pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token equivalent
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
#             attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
#         )

#     def resize_token_embeddings(self, new_num_tokens: int):
#         # Resize both input and output token embeddings
#         old_num_tokens, old_embedding_dim = self.model.embed_tokens.weight.shape
#         if new_num_tokens == old_num_tokens:
#             return

#         # Create new embedding
#         new_embedding = torch.nn.Embedding(new_num_tokens, old_embedding_dim).to(self.model.embed_tokens.weight.device)

#         # Copy existing weights
#         num_to_copy = min(old_num_tokens, new_num_tokens)
#         new_embedding.weight.data[:num_to_copy] = self.model.embed_tokens.weight.data[:num_to_copy]

#         self.model.embed_tokens = new_embedding
#         self.vocab_size = new_num_tokens



import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.distributed as dist

from tp_Llama_modules import TensorParallelLlamaModel


class TPLlamaForSequenceClassification(nn.Module):
    """
    Wraps the TensorParallelLlamaModel for sequence classification.
    We load a pretrained LlamaForSequenceClassification, extract its base LlamaModel
    and classification head, then build a TP backbone + a shared, single-rank classifier head.
    """

    def __init__(self,
                 model_name: str,
                 num_labels: int,
                 world_size: int,
                 group: torch.distributed.ProcessGroup,
                 local_rank: int):
        super().__init__()
        self.world_size = world_size
        self.group = group
        self.local_rank = local_rank

        # 1) Load the pretrained HF checkpoint
        self.hf_model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.config = self.hf_model.config

        # 2) Extract the base LlamaModel (without the classification head)
        pretrained_base = self.hf_model.llama  # LlamaModel

        # 3) Build our TP backbone
        self.tp_backbone = TensorParallelLlamaModel(
            config=pretrained_base.config,
            world_size=world_size,
            group=group,
            pretrained_model=pretrained_base
        )

        # 4) Build a shared classification head: linear(H → num_labels)
        #    We do NOT shard this head; we keep a full copy on each rank, but we will sync gradients.
        self.classifier = nn.Linear(self.config.hidden_size, num_labels, bias=True)
        # Copy weights & bias from HF model
        with torch.no_grad():
            self.classifier.weight.data.copy_(
                self.hf_model.classifier.weight.data.clone()
            )
            self.classifier.bias.data.copy_(
                self.hf_model.classifier.bias.data.clone()
            )

        # 5) If local_rank ≠ 0, we still keep a local copy of the classifier (for gradient sync).
        #    We'll manually all-reduce classifier grads in train_step.

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                **kwargs):
        """
        input_ids: [batch, seq]
        attention_mask: [batch, seq]
        labels: [batch]
        position_ids: [batch, seq] (for rotary)

        Returns: SequenceClassifierOutput(loss=?, logits=[batch, num_labels], ...)
        """
        # 1) Pass tokens through TP backbone → hidden_states: [batch, seq, H]
        outputs = self.tp_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            output_attentions=False
        )
        hidden_states = outputs[0]  # [batch, seq, H]

        # 2) Use the hidden-state of the first token as the pooled representation
        pooled_output = hidden_states[:, 0, :]  # [batch, H]

        # 3) Classifier head (full rank)
        logits = self.classifier(pooled_output)  # [batch, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        If you add tokens (e.g. a new pad token), resize both the embedding and
        the classifier head’s weight to match. We only handle embedding here.
        """
        old_embeddings = self.tp_backbone.embed_tokens
        old_num, old_dim = old_embeddings.weight.shape
        if new_num_tokens == old_num:
            return

        # Create new embedding
        new_embedding = nn.Embedding(new_num_tokens, old_dim).to(old_embeddings.weight.device)
        # Copy over existing weights
        new_embedding.weight.data[:min(old_num, new_num_tokens), :] = old_embeddings.weight.data[:min(old_num, new_num_tokens), :]
        self.tp_backbone.embed_tokens = new_embedding

        # If one wants to tie classifier to embeddings, handle that here.

