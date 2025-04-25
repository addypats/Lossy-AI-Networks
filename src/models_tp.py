import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from deepspeed.runtime.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from comms import LossyNetwork

class TPClassifier(nn.Module):
    def __init__(self, config, num_labels=2, loss_rate=0.001, seed=1234):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.linear1 = ColumnParallelLinear(config.hidden_size, config.intermediate_size)
        self.activation = nn.ReLU()
        self.linear2 = RowParallelLinear(config.intermediate_size, num_labels)

        # Integrate LossyNetwork for weight degradation simulation
        self.lossy_network = LossyNetwork(loss_rate=loss_rate)
        self.lossy_network.set_seed(seed)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Apply lossy communication simulation to parameters
        with torch.no_grad():
            for name, param in self.named_parameters():
                mask = self.lossy_network.send(param.data)
                param.data.copy_(self.lossy_network.receive(param.data, mask))

        x = self.embed(input_ids)
        x = x.mean(dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        logits = self.linear2(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

def get_classifier_and_tokenizer(model_name, num_labels=2, loss_rate=0.001, seed=1234):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id

    model = TPClassifier(config=config, num_labels=num_labels, loss_rate=loss_rate, seed=seed)
    return model, tokenizer