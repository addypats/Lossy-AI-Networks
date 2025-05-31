from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering

def get_classifier_and_tokenizer(model_name, num_labels=2, num_unfrozen_layers=None):
    """
    Load the model and tokenizer from Hugging Face Hub.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if num_unfrozen_layers is not None:
        for param in model.parameters():
            param.requires_grad = False
    
        for layer in model.model.layers[-num_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        model.score.requires_grad = True
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        
    return model, tokenizer