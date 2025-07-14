#!/usr/bin/env python3
"""
Prepare MNLI data for Megatron training
"""
import os
import json
from datasets import load_dataset

def prepare_mnli_data(output_dir):
    """Convert MNLI to text format for Megatron"""
    
    # Load MNLI dataset
    dataset = load_dataset("glue", "mnli")
    
    train_data = dataset["train"]
    valid_data = dataset["validation_matched"]
    
    # Prepare training data
    train_file = os.path.join(output_dir, "mnli_train.jsonl")
    with open(train_file, 'w') as f:
        for example in train_data:
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            label = example["label"]
            
            # Format: premise [SEP] hypothesis
            text = f"{premise} [SEP] {hypothesis}"
            
            # Create JSONL entry
            entry = {
                "text": text,
                "label": label
            }
            f.write(json.dumps(entry) + "\n")
    
    # Prepare validation data
    valid_file = os.path.join(output_dir, "mnli_valid.jsonl")
    with open(valid_file, 'w') as f:
        for example in valid_data:
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            label = example["label"]
            
            text = f"{premise} [SEP] {hypothesis}"
            
            entry = {
                "text": text,
                "label": label
            }
            f.write(json.dumps(entry) + "\n")
    
    print(f"Prepared MNLI data:")
    print(f"  Training: {train_file} ({len(train_data)} examples)")
    print(f"  Validation: {valid_file} ({len(valid_data)} examples)")

if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    prepare_mnli_data(output_dir)
