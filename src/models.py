from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def get_classifier_and_tokenizer(model_name, num_labels=2):
    """
    Load the model and tokenizer from Hugging Face Hub.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        
    return model, tokenizer