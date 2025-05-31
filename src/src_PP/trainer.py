from transformers import Trainer, TrainerCallback
from torch.distributed.pipeline.sync import Pipe
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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


class DistributedTrainerWithPipe(Trainer):
    def __init__(self, num_nodes, network, model, *args, **kwargs):
        self.lossy_network = network
        self.num_nodes = num_nodes

        # Pipeify model
        layers = model.model.layers
        num_stages = num_nodes
        layers_per_stage = len(layers) // num_stages

        device_ids = [f"cuda:{i}" for i in range(num_stages)]

        stage_modules = []
        stage_modules.append(nn.Sequential(
            model.model.embed_tokens.to(device_ids[0]),
            nn.Sequential(*layers[:layers_per_stage]).to(device_ids[0])
        ))

        for i in range(1, num_stages - 1):
            stage_modules.append(nn.Sequential(
                nn.Sequential(*layers[i * layers_per_stage:(i + 1) * layers_per_stage]).to(device_ids[i])
            ))

        stage_modules.append(nn.Sequential(
            nn.Sequential(*layers[-layers_per_stage:]).to(device_ids[-1]),
            model.model.norm.to(device_ids[-1]),
        ))

        pipeline = Pipe(nn.Sequential(*stage_modules), chunks=num_nodes, checkpoint='never')
        classifier = model.score.to(device_ids[-1])

        self.pipeline = pipeline
        self.classifier = classifier
        self.device_ids = device_ids

        model.forward = self.forward_with_pipe  # override default forward
        super().__init__(model=model, *args, **kwargs)

    def forward_with_pipe(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.pipeline(input_ids)
        out = apply_lossy_hook(out, self.lossy_network)
        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"].to(self.device_ids[0])
        attention_mask = inputs["attention_mask"].to(self.device_ids[0])
        labels = inputs["labels"].to(self.device_ids[-1])

        hidden_states = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = hidden_states[:, 0, :]  # Assume CLS token
        logits = self.classifier(cls_token)
        loss = F.cross_entropy(logits, labels)

        return (loss, logits) if return_outputs else loss


def compute_classfication_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


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
