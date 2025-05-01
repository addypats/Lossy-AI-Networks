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
        
        # default ring neighbors: each node sees (i–1)%N and (i+1)%N
        self.neighbors = {
            i: [ (i - 1) % num_nodes, (i + 1) % num_nodes ]
            for i in range(num_nodes)
        }
        
        # fill backup_weights with the model weights
        for name, param in self.model.named_parameters():
            self.backup_weights[name] = param.data.clone()


    def training_step(self, model, inputs, num_items_in_batch=None):
        # how many items total, per node
        batch_size = len(next(iter(inputs.values())))
        minibatch = batch_size // self.num_nodes

        # 1) Backup weights & clear grads
        for name, p in model.named_parameters():
            self.backup_weights[name] = p.data.clone()
            if p.grad is not None:
                p.grad.zero_()

        total_loss = torch.tensor(0.0, device=model.device, 
                                  dtype=torch.float16 if self.args.fp16 else torch.float32)

        # 2) Each node computes its local gradient
        local_grads = []
        for node in range(self.num_nodes):
            # restore model to the same starting point
            for name, p in model.named_parameters():
                p.data = self.backup_weights[name].clone()
                p.grad = None

            # split the batch
            split_inputs = {
                k: v[node * minibatch : (node + 1) * minibatch]
                for k, v in inputs.items()
            }

            # forward + backward
            loss = self.compute_loss(model, split_inputs)
            if self.args.fp16:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            total_loss += loss.detach()

            # snapshot grads
            grads_i = {
                name: p.grad.clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            local_grads.append(grads_i)

        # 3) Peer-to-peer exchange on our ring
        exchanged_grads = []
        for i in range(self.num_nodes):
            # start with own grad
            agg = {name: local_grads[i][name].clone() for name in local_grads[i]}
            # incorporate neighbors
            for nbr in self.neighbors[i]:
                for name in agg:
                    mask = self.network.send(local_grads[nbr][name])
                    recv = self.network.receive(local_grads[nbr][name], mask)
                    agg[name] += recv
            # average over (self + neighbors)
            for name in agg:
                agg[name] /= (1 + len(self.neighbors[i]))
            exchanged_grads.append(agg)

        # 4) Build a final gradient by averaging all node-wise gradients
        final_grad = {}
        for name in exchanged_grads[0]:
            # sum over nodes
            s = sum(exchanged_grads[n][name] for n in range(self.num_nodes))
            final_grad[name] = s / self.num_nodes

        # 5) Load that into the model and restore weights
        for name, p in model.named_parameters():
            if name in final_grad:
                p.grad = final_grad[name]
            # reset to original before the optimizer step
            p.data = self.backup_weights[name].clone()

        # Trainer will then do optimizer.step() and scheduler.step()
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
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)

    def on_evaluate(self, args, state, control, **kwargs):
        
        accuracy = kwargs["metrics"]["eval_accuracy"]
        if accuracy > self.args['target_acc']:
            print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
            control.should_training_stop = True

        for ac in self.args['report_ttac']: # since it is sorted in descending order we only report the last one reached
            if accuracy >= ac:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {ac},  Step: {state.global_step}\n")
                break
        return super().on_evaluate(args, state, control, **kwargs)
