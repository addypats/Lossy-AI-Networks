import argparse
import math
import torch
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from oslo import ParallelContext, ParallelMode
from oslo.torch.nn.parallel import TensorParallel
from oslo.transformers import OSOParallelPlugin
from datasets import load_dataset

class LossyNetwork:
    def __init__(self, loss_rate: float = 0.001):
        self.loss_rate = loss_rate

    def set_seed(self, seed: int):
        self.seed = seed

    def send(self, data: torch.Tensor) -> torch.Tensor:
        data_size = data.numel()
        float_size = data.element_size()
        num_bytes = data_size * float_size
        num_packets = math.ceil(num_bytes / 1450)
        packets_mask = torch.rand(num_packets) > self.loss_rate
        return packets_mask
    
    def receive(self, data: torch.Tensor, packets_mask: torch.Tensor) -> torch.Tensor:
        if packets_mask.all():
            return data
        num_packets = len(packets_mask)
        number_per_packet = 1450 // data.element_size() + 1
        flat = data.flatten()
        indices = torch.arange(num_packets * number_per_packet, device=data.device)
        indices = indices[indices < flat.numel()]
        mask = packets_mask.repeat_interleave(number_per_packet)[:indices.numel()]
        flat[~mask] = 0.0
        return flat.view_as(data)

class LossyTrainer(Trainer):
    def __init__(self, loss_network=None, **kwargs):
        super().__init__(**kwargs)
        self.loss_network = loss_network or LossyNetwork()
        
    def training_step(self, model, inputs):
        inputs["input_ids"] = self._apply_network_loss(inputs["input_ids"])
        return super().training_step(model, inputs)
    
    def _apply_network_loss(self, tensor):
        if self.args.process_index == 0:
            packets_mask = self.loss_network.send(tensor.cpu())
        else:
            packets_mask = torch.empty(0)
        
        packets_mask = self._accelerator.broadcast(packets_mask, from_process=0)
        return self.loss_network.receive(tensor, packets_mask)

def get_dataset(args, tokenizer):
    # ... (include your dataset loading functions from paste.txt here)    
    if args.dataset == "sst2":
        return get_sst2(tokenizer, args)
    elif args.dataset == "cola":
        return get_cola(tokenizer, args)
    elif args.dataset == "mnli":
        return get_mnli(tokenizer, args)
    elif args.dataset == "winogrande":
        return get_winogrande(tokenizer, args)
    elif args.dataset == "arc":
        return get_arc(tokenizer, args)
    elif args.dataset == "hellaswag":
        return get_hellaswag(tokenizer, args)
    elif args.dataset == "piqa": 
        return get_piqa(tokenizer, args)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")


def get_winogrande(tokenizer, args):

    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    def preprocess(data):
        # Improve Winogrande preprocessing by formatting the input as a choice task
        # Replace the placeholder '_' with each option and tokenize separately
        sentence = data["sentence"]
        option1 = data["option1"]
        option2 = data["option2"]
        
        # Find the placeholder marker in the sentence
        if "_" in sentence:
            placeholder = "_"
        else:
            # Some versions might use a different placeholder
            placeholder = "___"
            
        # Create two complete sentences with each option
        sentence1 = sentence.replace(placeholder, option1)
        sentence2 = sentence.replace(placeholder, option2)
        
        # Use the correct option as the answer
        label = 0 if data["answer"] == '1' else 1
        
        # Tokenize for binary classification - encode each option separately
        encodings = tokenizer(
            [sentence1, sentence2],
            truncation=True,
            padding="max_length", 
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Select the encoding of the correct option
        return {
            'input_ids': encodings["input_ids"][label].tolist(),
            'attention_mask': encodings["attention_mask"][label].tolist(),
            'labels': label
        }
    train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])

    return train_dataset, eval_dataset

def get_sst2(tokenizer, args):

    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    max_length = args.max_length if args.max_length > 0 else 128

    def preprocess(data):
        return {
            'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': data["label"]
        }
        
    train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    
    return train_dataset, eval_dataset


def get_cola(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 128
    dataset = load_dataset("glue", "cola")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    def preprocess(data):
        return {
            'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': data["label"]
        }
        
    train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])

    return train_dataset, eval_dataset

def get_mnli(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 128
    dataset = load_dataset("glue", "mnli")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_matched"]  # Using matched validation set

    def preprocess(data):
        return {
            'input_ids': tokenizer(data["premise"], data["hypothesis"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(data["premise"], data["hypothesis"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': data["label"]
        }
        
    train_dataset = train_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])

    return train_dataset, eval_dataset

def get_arc(tokenizer, args):

    keys = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3,
    }
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    def preprocess(data):

        question = data['question']
        choices = list(zip(data['choices']['text'], data['choices']['label']))
        
        choices = ' '.join([f"{label}: {text}" for text, label in choices])
        question = f"{question}\n\n{choices}"
        return {
            'input_ids': tokenizer(question, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(question, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': keys.get(data["answerKey"], -1)
        }
        
    train_dataset = train_dataset.map(preprocess, remove_columns=["question", "answerKey", "choices"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["question", "answerKey", "choices"])

    # Filter out invalid labels
    train_dataset = train_dataset.filter(lambda x: x['labels'] != -1)
    eval_dataset = eval_dataset.filter(lambda x: x['labels'] != -1)

    return train_dataset, eval_dataset

def get_hellaswag(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("hellaswag")
    
    # Limit dataset size if specified
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    if args.max_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    
    def preprocess(data):
        # Format context and endings for multiple choice
        context = data["ctx"]
        endings = data["endings"]
        
        # Format as multiple choice
        choices = [f"{context} {ending}" for ending in endings]
        
        # Create encodings for all choices
        encodings = tokenizer(
            choices,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Get label (correct ending index)
        label = int(data["label"])
        
        return {
            'input_ids': encodings["input_ids"][label].tolist(),
            'attention_mask': encodings["attention_mask"][label].tolist(),
            'labels': label
        }
    
    # Map preprocessing function to datasets
    train_dataset = train_dataset.map(preprocess, remove_columns=["ctx", "endings", "label", "activity_label", "source_id"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["ctx", "endings", "label", "activity_label", "source_id"])
    
    return train_dataset, eval_dataset


def get_piqa(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("piqa", trust_remote_code=True)
    
    def preprocess(data):
        
        goal = data["goal"]
        sol1 = data['sol1']
        sol2 = data['sol2']
                
        input = "Which of the following is a better solution to the problem?\n\n" + f" 1) {goal} {sol1} \n 2) {goal} {sol2}"
        label = data["label"]

        
        return {
            'input_ids': tokenizer(input, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(input, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': label
        }
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    train_dataset = train_dataset.map(preprocess, remove_columns=["goal", "sol1", "sol2", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["goal", "sol1", "sol2", "label"])
    
    return train_dataset, eval_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="gpt2-medium")
    parser.add_argument("--loss_rate", type=float, default=0.01)
    parser.add_argument("--tensor_parallel", type=int, default=4)
    args = parser.parse_args()

    # Initialize parallel context
    parallel_context = ParallelContext.from_torch(
        tensor_parallel_size=args.tensor_parallel,
        tensor_parallel_mode=ParallelMode.TENSOR_1D
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    # Apply tensor parallelism
    model = TensorParallel(model, parallel_context)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        fp16=True,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        logging_steps=100,
        save_steps=500,
        deepspeed="ds_config.json",
        plugins=[OSOParallelPlugin(tensor_parallel_size=args.tensor_parallel)]
    )

    # Initialize trainer
    trainer = LossyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_network=LossyNetwork(loss_rate=args.loss_rate)
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
