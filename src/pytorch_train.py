import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os
import time
from comms import LossyNetwork
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score

def train_step(model, inputs, optimizer, num_nodes, network):
    """Single training step with simulated distributed training and packet loss."""
    model.train()
    
    # Get batch size from inputs
    batch_size = len(inputs['input_ids'])
    minibatch_size = batch_size // num_nodes
    
    # Initialize gradient accumulation
    averaged_gradients = {k: torch.zeros_like(v, device=v.device) 
                         for k, v in model.named_parameters() if v.requires_grad}
    
    total_loss = 0.0
    
    # Simulate multi-node training
    for i in range(num_nodes):
        # Get per-node batch slice
        start_idx = i * minibatch_size
        end_idx = min((i + 1) * minibatch_size, batch_size)
        
        if start_idx >= batch_size:
            continue
            
        # Extract minibatch for this node
        node_inputs = {
            'input_ids': inputs['input_ids'][start_idx:end_idx].to(model.device),
            'attention_mask': inputs['attention_mask'][start_idx:end_idx].to(model.device),
            'labels': inputs['labels'][start_idx:end_idx].to(model.device)
        }
        
        # Forward pass
        outputs = model(**node_inputs)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Simulate packet loss in gradient communication
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Apply packet loss to gradients
                mask = network.send(param.grad)
                received_grad = network.receive(param.grad, mask)
                averaged_gradients[name] += received_grad
        
        # Add to total loss
        total_loss += loss.item()
        
        # Clear gradients for next node
        model.zero_grad()
    
    # Average the gradients across nodes
    for name, grad in averaged_gradients.items():
        averaged_gradients[name] = grad / num_nodes
    
    # Apply averaged gradients to model
    for name, param in model.named_parameters():
        if param.requires_grad and name in averaged_gradients:
            param.grad = averaged_gradients[name]
    
    # Update model parameters
    optimizer.step()
    optimizer.zero_grad()
    
    return total_loss / num_nodes

def evaluate(model, eval_dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            logits = outputs.logits
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def train_to_accuracy(args):
    """Train model until target accuracy is reached."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup lossy network
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)
    
    # Load dataset
    if args.dataset == "winogrande":
        dataset = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)    
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        num_labels = 2
        
        # Apply sample limit if specified
        if args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(args.max_samples // 5, len(eval_dataset))))
            
        model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-large-uncased", 
            num_labels=num_labels
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

        def preprocess(data):
            # Improve Winogrande preprocessing by formatting the input as a choice task
            sentence = data["sentence"]
            option1 = data["option1"]
            option2 = data["option2"]
            
            # Find the placeholder marker in the sentence
            if "_" in sentence:
                placeholder = "_"
            else:
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
                max_length=256,
                return_tensors="pt"
            )
            
            # Select the encoding of the correct option
            return {
                'input_ids': encodings["input_ids"][label].tolist(),
                'attention_mask': encodings["attention_mask"][label].tolist(),
                'labels': label
            }
    
    elif args.dataset == "sst2":
        dataset = load_dataset("glue", "sst2")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        num_labels = 2
        
        # Apply sample limit if specified
        if args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(args.max_samples // 5, len(eval_dataset))))
            
        model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-large-uncased", 
            num_labels=num_labels
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
        
        def preprocess(data):
            return {
                'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=128)["input_ids"],
                'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=128)["attention_mask"],
                'labels': data["label"]
            }
    
    elif args.dataset == "cola":
        dataset = load_dataset("glue", "cola")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        num_labels = 2
        
        # Apply sample limit if specified
        if args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(args.max_samples // 5, len(eval_dataset))))
            
        model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-large-uncased", 
            num_labels=num_labels
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
        
        def preprocess(data):
            return {
                'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=128)["input_ids"],
                'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=128)["attention_mask"],
                'labels': data["label"]
            }
        
    elif args.dataset == "mnli":
        dataset = load_dataset("glue", "mnli")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation_matched"]  # Using matched validation set
        num_labels = 3
        
        # Apply sample limit if specified
        if args.max_samples > 0:
            train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
            eval_dataset = eval_dataset.select(range(min(args.max_samples // 5, len(eval_dataset))))
            
        model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-large-uncased", 
            num_labels=num_labels
        ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
        
        def preprocess(data):
            return {
                'input_ids': tokenizer(data["premise"], data["hypothesis"], 
                                      truncation=True, padding="max_length", max_length=128)["input_ids"],
                'attention_mask': tokenizer(data["premise"], data["hypothesis"], 
                                          truncation=True, padding="max_length", max_length=128)["attention_mask"],
                'labels': data["label"]
            }
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Preprocess datasets
    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)
    
    # Convert to PyTorch format
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size
    )
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    best_accuracy = 0.0
    no_improvement_count = 0
    global_step = 0
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Main training loop
    print(f"Training until reaching accuracy {args.target_accuracy} or maximum {args.max_steps} steps")
    print(f"Evaluating every {args.eval_steps} steps")
    
    while global_step < args.max_steps:
        epoch_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Step {global_step}"):
            # Train step
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = train_step(model, batch, optimizer, args.num_nodes, network)
            epoch_loss += loss
            global_step += 1
            
            # Evaluate if needed
            if global_step % args.eval_steps == 0:
                accuracy = evaluate(model, eval_dataloader, device)
                elapsed_time = time.time() - start_time
                
                print(f"Step {global_step} | Accuracy: {accuracy:.4f} | Time: {elapsed_time:.2f}s")
                
                # Save results
                with open(os.path.join(args.output_dir, "results.txt"), "a") as f:
                    f.write(f"{global_step},{accuracy:.4f},{elapsed_time:.2f}\n")
                
                # Check if target accuracy reached
                if accuracy >= args.target_accuracy:
                    print(f"Target accuracy {args.target_accuracy} reached at step {global_step}!")
                    # Save final model
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
                    return {
                        "steps_to_accuracy": global_step,
                        "final_accuracy": accuracy,
                        "time_to_accuracy": elapsed_time
                    }
                
                # Early stopping logic
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    no_improvement_count = 0
                    # Save best model
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_best.pt"))
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= args.patience:
                    print(f"No improvement for {args.patience} evaluations. Stopping training.")
                    return {
                        "steps_to_accuracy": global_step,
                        "final_accuracy": best_accuracy,
                        "time_to_accuracy": elapsed_time,
                        "reached_target": False
                    }
            
            # Stop if max steps reached
            if global_step >= args.max_steps:
                break
    
    # Training completed without reaching target
    accuracy = evaluate(model, eval_dataloader, device)
    elapsed_time = time.time() - start_time
    
    print(f"Maximum steps {args.max_steps} reached. Final accuracy: {accuracy:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))
    
    return {
        "steps_to_accuracy": args.max_steps,
        "final_accuracy": accuracy,
        "time_to_accuracy": elapsed_time,
        "reached_target": False
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train to Accuracy with PyTorch and Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', type=str, default='winogrande', choices=['winogrande', 'sst2', 'cola', 'mnli'], 
                        help='Dataset to use for training')
    parser.add_argument('--max_samples', type=int, default=0, 
                        help='Maximum number of training samples to use (0 for all)')
    parser.add_argument('--target_accuracy', type=float, default=0.75, 
                        help='Target accuracy to train until')
    parser.add_argument('--eval_steps', type=int, default=100, 
                        help='Evaluate every N steps')
    parser.add_argument('--patience', type=int, default=3, 
                        help='Early stopping patience (number of evaluations with no improvement)')
    parser.add_argument('--max_steps', type=int, default=100000, 
                        help='Maximum number of training steps')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Train to accuracy
    results = train_to_accuracy(args)
    
    # Print final results
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value}") 