from transformers import Trainer, TrainingArguments, TrainerCallback
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class DistributedTrainer(Trainer):

    def __init__(self, num_nodes, network, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.network = network
        self.backup_weights = {}
        # fill backup_weights with the model weights
        for name, param in self.model.named_parameters():
            self.backup_weights[name] = param.data.clone()


    def training_step(self, model, inputs, num_items_in_batch=0):

        num_items_in_batch = len(inputs[list(inputs.keys())[0]])
        minibatch_size = num_items_in_batch // self.num_nodes
        averaged_gradients = {k:torch.zeros_like(v, device=model.device) for k, v in model.named_parameters() if v.requires_grad}
        
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device, dtype=torch.float16 if self.args.fp16 else torch.float32)
        for name, param in model.named_parameters():
            self.backup_weights[name] = param.data.clone()    

        for i in range(self.num_nodes):
            
            for name, param in model.named_parameters():
                mask = self.network.send(param.data)
                param.data = self.network.receive(param.data, mask) ## This simulates the packet loss during broadcasting 

            inputs_split = {k: v[i * minibatch_size:(i + 1) * minibatch_size] for k, v in inputs.items()}
            
            # Let Transformers Trainer handle FP16 internally
            loss = self.compute_loss(model, inputs_split)
            
            # Handle fp16 properly by using the accelerator for backward pass
            if self.args.fp16:
                self.accelerator.backward(loss)
            else:
                loss.backward()
                
            total_loss = total_loss + loss.detach()  # Add to total_loss for reporting

            for name,param in model.named_parameters():
                if param.grad is not None:  # Check if grad exists to avoid NoneType errors
                    mask = self.network.send(param.grad)
                    averaged_gradients[name] = averaged_gradients[name] + self.network.receive(param.grad, mask)

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.grad = averaged_gradients[name]

        for name, param in model.named_parameters():
            param.data = self.backup_weights[name]  # Restore the original weights
        
        return total_loss / self.num_nodes


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
        # ensure thresholds are in descending order
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)

    def on_evaluate(self, args, state, control, **kwargs):
        # grab whichever key exists: "eval_accuracy" OR "accuracy"
        metrics = kwargs.get("metrics", {})
        accuracy = metrics.get("eval_accuracy", metrics.get("accuracy"))
        if accuracy is None:
            # nothing to do if accuracy isn’t in metrics
            return super().on_evaluate(args, state, control, **kwargs)

        if accuracy > self.args['target_acc']:
            print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
            control.should_training_stop = True

        # report the highest threshold you've passed
        for ac in self.args['report_ttac']:
            if accuracy >= ac:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {ac}, Step: {state.global_step}\n")
                break

        return super().on_evaluate(args, state, control, **kwargs)
