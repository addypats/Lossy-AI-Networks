# # from transformers import AutoModelForSequenceClassification, AutoTokenizer
# # import torch

# # def get_classifier_and_tokenizer(model_name, num_labels=2):
# #     """
# #     Load the model and tokenizer from Hugging Face Hub.
# #     """
# #     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
# #     tokenizer = AutoTokenizer.from_pretrained(model_name)
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #         model.resize_token_embeddings(len(tokenizer))
# #         model.config.pad_token_id = tokenizer.pad_token_id
        
# #     return model, tokenizer




# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# def get_classifier_and_tokenizer(model_name, num_labels=2, num_unfrozen_layers=None):
#     """
#     Load the model and tokenizer from Hugging Face Hub.
#     """
#     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

#     if num_unfrozen_layers is not None:
#         for param in model.parameters():
#             param.requires_grad = False
    
#         for layer in model.model.layers[-num_unfrozen_layers:]:
#             for param in layer.parameters():
#                 param.requires_grad = True
#         model.score.requires_grad = True
        
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         model.resize_token_embeddings(len(tokenizer))
#         model.config.pad_token_id = tokenizer.pad_token_id
        
#     return model, tokenizer


# def get_qa_model_and_tokenizer(model_name, num_unfrozen_layers=None):

#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.add_special_tokens({'pad_token': '|<pad>|'})
#         model.resize_token_embeddings(len(tokenizer))
#         model.config.pad_token_id = tokenizer.pad_token_id
#     else:
#         print(f"Using existing pad token: {tokenizer.pad_token}")

#     tokenizer.padding_side = "left"
#     model.generation_config.top_p = None
#     model.generation_config.temperature = None    
#     return model, tokenizer


from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

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


def get_qa_model_and_tokenizer(model_name, num_unfrozen_layers=None):

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '|<pad>|'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        print(f"Using existing pad token: {tokenizer.pad_token}")

    tokenizer.padding_side = "left"
    model.generation_config.top_p = None
    model.generation_config.temperature = None
    model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer