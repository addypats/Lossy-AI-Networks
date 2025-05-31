from transformers import Trainer, TrainerCallback
from torch.distributed.pipeline.sync import Pipe
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# ======== Lossy Gradient Wrapper ========
class LossyGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, lossy_network):
        ctx.lossy_network = lossy_network
        ctx.save_for_backward(input_tensor)
        return input_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        lossy_net = ctx.lossy_network
        with torch.no_grad():
            packet_mask = lossy_net.send(grad_output)
            noisy_grad = lossy_net.receive(grad_output.clone(), packet_mask)
        return noisy_grad, None


def apply_lossy_hook(tensor, lossy_network):
    return LossyGradFunction.apply(tensor, lossy_network)


# ======== Trainer with Pipeline Parallelism ========
class DistributedTrainerWithPipe(Trainer):
    def __init__(self, num_nodes, network, model, *args, **kwargs):
        self.lossy_network = network
        self.num_nodes = num_nodes

        # Extract transformer layers
        layers = model.model.layers
        total_layers = len(layers)
        layers_per_stage = total_layers // num_nodes

        # Set up device IDs (cuda:0, cuda:1, ...)
        device_ids = [f"cuda:{i}" for i in range(num_nodes)]

        # Create pipeline stages
        stage_modules = []

        # First stage: embedding + layers
        stage_modules.append(nn.Sequential(
            model.model.embed_tokens.to(device_ids[0]),
            nn.Sequential(*layers[:layers_per_stage]).to(device_ids[0])
        ))

        # Middle stages
        for i in range(1, num_nodes - 1):
            stage_modules.append(nn.Sequential(
                nn.Sequential(*layers[i * layers_per_stage:(i + 1) * layers_per_stage]).to(device_ids[i])
            ))

        # Last stage: last layers + norm
        stage_modules.append(nn.Sequential(
            nn.Sequential(*layers[(num_nodes - 1) * layers_per_stage:]).to(device_ids[-1]),
            model.model.norm.to(device_ids[-1])
        ))

        # Wrap model with Pipe
        self.pipeline = Pipe(nn.Sequential(*stage_modules), chunks=num_nodes, checkpoint='never')

        # Move classifier head to final stage device
        self.classifier = model.score.to(device_ids[-1])
        self.device_ids = device_ids

        # Replace forward
        model.forward = self.forward_with_pipe
        super().__init__(model=model, *args, **kwargs)

    def forward_with_pipe(self, input_ids=None, attention_mask=None, **kwargs):
        return self.pipeline(input_ids)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"].to(self.device_ids[0])
        attention_mask = inputs["attention_mask"].to(self.device_ids[0])
        labels = inputs["labels"].to(self.device_ids[-1])

        # Forward through pipeline
        hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = apply_lossy_hook(hidden_states, self.lossy_network)

        cls_token = hidden_states[:, 0, :]  # CLS token
        logits = self.classifier(cls_token)
        loss = F.cross_entropy(logits, labels)

        return (loss, logits) if return_outputs else loss


# ======== Metrics Function ========
def compute_classfication_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


# ======== TTAC Reporting Callback ========
class MyClassifierCallback(TrainerCallback):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)

    def on_evaluate(self, args, state, control, **kwargs):
        accuracy = kwargs["metrics"].get("eval_accuracy", 0)
        if accuracy > self.args['target_acc']:
            print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
            control.should_training_stop = True

        for ac in self.args['report_ttac']:
            if accuracy >= ac:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {ac},  Step: {state.global_step}\n")
                break
        return super().on_evaluate(args, state, control, **kwargs)

