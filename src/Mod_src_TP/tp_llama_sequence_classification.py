import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class TPLlamaForSequenceClassification(nn.Module):
    def __init__(self, config, world_size, group):
        super().__init__()
        from tp_Llama_modules import TensorParallelLlamaModel
        self.config = config
        self.model = TensorParallelLlamaModel(config, world_size, group)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_dim]
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token equivalent
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    def resize_token_embeddings(self, new_num_tokens: int):
        # Resize both input and output token embeddings
        old_num_tokens, old_embedding_dim = self.model.embed_tokens.weight.shape
        if new_num_tokens == old_num_tokens:
            return

        # Create new embedding
        new_embedding = torch.nn.Embedding(new_num_tokens, old_embedding_dim).to(self.model.embed_tokens.weight.device)

        # Copy existing weights
        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_embedding.weight.data[:num_to_copy] = self.model.embed_tokens.weight.data[:num_to_copy]

        self.model.embed_tokens = new_embedding
        self.vocab_size = new_num_tokens

