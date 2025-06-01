# trainer.py - Modified for Pipeline Parallelism
from transformers import Trainer, TrainerCallback
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
from typing import Dict, Any, Optional, List
import math

class PipelineStage(nn.Module):
    """Wrapper for pipeline stages that applies lossy network simulation"""
   
    def __init__(self, module, network, stage_id, is_last_stage=False):
        super().__init__()
        self.module = module
        self.network = network
        self.stage_id = stage_id
        self.is_last_stage = is_last_stage
        self.backup_weights = {}
        self.device = None  # Will be set when moved to device
       
        # Store backup weights for this stage
        for name, param in self.module.named_parameters():
            self.backup_weights[name] = param.data.clone()
    
    def to(self, device):
        """Override to method to properly handle device placement"""
        super().to(device)
        self.device = device
        self.module = self.module.to(device)
        
        # Move backup weights to the same device
        for name in self.backup_weights:
            self.backup_weights[name] = self.backup_weights[name].to(device)
        
        return self
   
    def forward(self, x):
        # Ensure input is on the correct device
        if self.device is not None:
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
            elif isinstance(x, dict):
                x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in x.items()}
        
        output = self.module(x)
       
        # Apply lossy network to gradients during backward pass
        if output.requires_grad and not self.is_last_stage:
            output.register_hook(self.gradient_hook)
           
        return output
   
    def gradient_hook(self, grad):
        """Apply lossy network to gradients"""
        if grad is None:
            return grad
           
        # Apply packet loss to gradients
        mask = self.network.send(grad)
        lossy_grad = self.network.receive(grad, mask)
       
        return lossy_grad
   
    def apply_weight_loss(self):
        """Apply lossy network to weights (for broadcasting simulation)"""
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                # Store backup
                self.backup_weights[name] = param.data.clone()
               
                # Apply packet loss
                mask = self.network.send(param.data)
                param.data = self.network.receive(param.data, mask)
   
    def restore_weights(self):
        """Restore original weights from backup"""
        for name, param in self.module.named_parameters():
            if name in self.backup_weights:
                param.data = self.backup_weights[name].clone()


class PipelineParallelModel(nn.Module):
    """Pipeline parallel model that splits layers across devices"""
    
    def __init__(self, base_model, network, num_stages=None):
        super().__init__()
        self.network = network
        self.num_stages = num_stages or torch.cuda.device_count()
        self.devices = [f'cuda:{i}' for i in range(min(self.num_stages, torch.cuda.device_count()))]
        
        # Split model first, then move to devices
        self.stages = self._split_model(base_model)
        
        # Move stages to devices AFTER creating them
        for i, stage in enumerate(self.stages):
            device_idx = i % len(self.devices)
            stage.to(self.devices[device_idx])
            print(f"Stage {i} moved to {self.devices[device_idx]}")
    
    def _split_model(self, model):
        """Split model into pipeline stages"""
        stages = []
        
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # For Llama-like models
            embed_tokens = model.model.embed_tokens if hasattr(model.model, 'embed_tokens') else None
            layers = model.model.layers
            norm = model.model.norm if hasattr(model.model, 'norm') else None
            classifier = model.score if hasattr(model, 'score') else model.classifier
            
            total_layers = len(layers)
            layers_per_stage = max(1, total_layers // (self.num_stages - 2)) if self.num_stages > 2 else total_layers
            
            # Stage 0: Embedding
            if embed_tokens is not None:
                stage_0 = PipelineStage(embed_tokens, self.network, 0)
                stages.append(stage_0)
            
            # Middle stages: Transformer layers
            current_stage = 1 if embed_tokens is not None else 0
            for i in range(0, total_layers, layers_per_stage):
                end_idx = min(i + layers_per_stage, total_layers)
                stage_layers = nn.Sequential(*layers[i:end_idx])
                stage = PipelineStage(stage_layers, self.network, current_stage)
                stages.append(stage)
                current_stage += 1
            
            # Final stage: Norm + Classifier
            final_modules = []
            if norm is not None:
                final_modules.append(norm)
            final_modules.append(classifier)
            
            final_stage = PipelineStage(
                nn.Sequential(*final_modules), 
                self.network, 
                current_stage, 
                is_last_stage=True
            )
            stages.append(final_stage)
            
        else:
            # Fallback: split sequential layers
            if hasattr(model, 'features'):
                features = list(model.features.children())
            else:
                features = list(model.children())
            
            layers_per_stage = max(1, len(features) // self.num_stages)
            
            for i in range(0, len(features), layers_per_stage):
                end_idx = min(i + layers_per_stage, len(features))
                stage_layers = nn.Sequential(*features[i:end_idx])
                is_last = (end_idx == len(features))
                stage = PipelineStage(stage_layers, self.network, i // layers_per_stage, is_last)
                stages.append(stage)
        
        return nn.ModuleList(stages)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass through pipeline stages"""
        # Handle different input formats
        if input_ids is not None:
            # Standard HuggingFace format - use input_ids as the main input
            x = input_ids
        elif len(kwargs) == 1 and isinstance(list(kwargs.values())[0], torch.Tensor):
            # Single tensor passed as kwarg
            x = list(kwargs.values())[0]
        elif isinstance(kwargs.get('inputs'), torch.Tensor):
            # Tensor passed as 'inputs' kwarg
            x = kwargs['inputs']
        else:
            # Fallback - try to find the first tensor in kwargs
            tensor_values = [v for v in kwargs.values() if isinstance(v, torch.Tensor)]
            if tensor_values:
                x = tensor_values[0]
            else:
                raise ValueError(f"No valid tensor input found. Got: input_ids={input_ids}, kwargs keys: {list(kwargs.keys())}")
    
        # Forward pass through pipeline stages
        for i, stage in enumerate(self.stages):
            # The stage will handle device placement internally
            x = stage(x)
            # Debug output
            if isinstance(x, torch.Tensor):
                print(f"After stage {i}: tensor on {x.device}")
            # If we need to move to next stage's device for next iteration
            # (this is handled by the next stage's forward method)
    
        # Return in the format expected by HuggingFace
        # If labels were provided, this is training mode
        if labels is not None:
            # For training, return a dict with loss and logits
            # You may need to compute loss here or let the trainer handle it
            return {
                'logits': x,
                'labels': labels  # Pass through labels for loss computation
            }
        else:
            # For inference, just return logits
            return {'logits': x}

    def apply_weight_loss_all_stages(self):
        """Apply weight loss to all stages"""
        for stage in self.stages:
            stage.apply_weight_loss()
    
    def restore_weights_all_stages(self):
        """Restore weights for all stages"""
        for stage in self.stages:
            stage.restore_weights()

class DistributedTrainer(Trainer):
    def __init__(self, num_nodes, network, *args, **kwargs):
        # Extract model before calling super().__init__
        model = kwargs.get('model')
        
        # Convert to pipeline parallel model
        self.original_model = model
        pipeline_model = PipelineParallelModel(model, network, num_stages=num_nodes)
        kwargs['model'] = pipeline_model
        
        super().__init__(*args, **kwargs)
        self.num_nodes = num_nodes
        self.network = network
        self.pipeline_model = pipeline_model
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Modified training step for pipeline parallelism"""
        
        if num_items_in_batch is None:
            num_items_in_batch = len(inputs[list(inputs.keys())[0]])
        
        minibatch_size = max(1, num_items_in_batch // self.num_nodes)
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device, 
                                dtype=torch.float16 if self.args.fp16 else torch.float32)
        
        # Store gradients across micro-batches
        accumulated_gradients = {}
        
        for i in range(self.num_nodes):
            # Apply weight loss (simulating broadcasting)
            # model.apply_weight_loss_all_stages()
            if hasattr(model, 'module'):
                # Model is wrapped in DDP
                model.module.apply_weight_loss_all_stages()
            else:
                # Model is not wrapped in DDP
                model.apply_weight_loss_all_stages()
            
            # Create micro-batch
            start_idx = i * minibatch_size
            end_idx = min((i + 1) * minibatch_size, num_items_in_batch)
            
            if start_idx >= num_items_in_batch:
                break
                
            inputs_split = {k: v[start_idx:end_idx] for k, v in inputs.items()}
            
            # Forward and backward pass
            model.zero_grad()
            loss = self.compute_loss(model, inputs_split)
            
            if self.args.fp16:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            
            total_loss += loss.detach()
            
            # Accumulate gradients with lossy network simulation
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Apply packet loss to gradients
                    mask = self.network.send(param.grad)
                    lossy_grad = self.network.receive(param.grad, mask)
                    
                    if name not in accumulated_gradients:
                        accumulated_gradients[name] = torch.zeros_like(param.grad)
                    accumulated_gradients[name] += lossy_grad
            
            # Restore original weights
            model.restore_weights_all_stages()
        
        # Set averaged gradients
        for name, param in model.named_parameters():
            if name in accumulated_gradients:
                param.grad = accumulated_gradients[name] / self.num_nodes
        
        return total_loss / self.num_nodes
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss for pipeline model"""
        
        # Get input_ids and move to first device
        input_ids = inputs.get('input_ids')
        if input_ids is not None:
            # Move to first stage device
            first_device = model.devices[0] if hasattr(model, 'devices') else 'cuda:0'
            input_ids = input_ids.to(first_device)
        
        # Forward pass
        if hasattr(model, 'pipeline_model'):
            outputs = model(input_ids)
        else:
            outputs = model(**inputs)
        
        # Extract logits and labels
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        labels = inputs.get('labels')
        if labels is not None:
            labels = labels.to(logits.device)
            
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return (loss, outputs) if return_outputs else loss

def compute_classfication_metrics(eval_pred):
    """Same as original"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

class MyClassifierCallback(TrainerCallback):
    """Same as original"""
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)
    
    def on_evaluate(self, args, state, control, **kwargs):
        accuracy = kwargs["metrics"]["eval_accuracy"]
        if accuracy > self.args['target_acc']:
            print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
            control.should_training_stop = True
        
        for ac in self.args['report_ttac']:
            if accuracy >= ac:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {ac}, Step: {state.global_step}\n")
                break
        
        return super().on_evaluate(args, state, control, **kwargs)
