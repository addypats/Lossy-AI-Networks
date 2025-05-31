from datasets import load_dataset

def get_dataset(args, tokenizer):
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