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

def format_prompt(prompt):
    return f"[INST] {prompt} [/INST]"

def get_winogrande(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    def preprocess(data):
        sentence = data["sentence"]
        option1 = data["option1"]
        option2 = data["option2"]
        placeholder = "_" if "_" in sentence else "___"
        sentence1 = sentence.replace(placeholder, option1)
        sentence2 = sentence.replace(placeholder, option2)
        prompt = f"Choose the better option:\n1. {sentence1}\n2. {sentence2}"
        prompt = format_prompt(prompt)
        label = 0 if data["answer"] == '1' else 1
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        return {
            'input_ids': encoding["input_ids"],
            'attention_mask': encoding["attention_mask"],
            'labels': label
        }

    train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])
    return train_dataset, eval_dataset

def get_piqa(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("piqa", trust_remote_code=True)

    def preprocess(data):
        goal = data["goal"]
        sol1 = data['sol1']
        sol2 = data['sol2']
        label = data['label']
        prompt = f"Which of the following solutions best solves the problem?\nProblem: {goal}\n1) {sol1}\n2) {sol2}"
        prompt = format_prompt(prompt)
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        return {
            'input_ids': encoding["input_ids"],
            'attention_mask': encoding["attention_mask"],
            'labels': label
        }

    train_dataset = dataset["train"].map(preprocess, remove_columns=["goal", "sol1", "sol2", "label"])
    eval_dataset = dataset["validation"].map(preprocess, remove_columns=["goal", "sol1", "sol2", "label"])
    return train_dataset, eval_dataset

def get_mnli(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("glue", "mnli")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation_matched"]

    def preprocess(data):
        premise = data["premise"]
        hypothesis = data["hypothesis"]
        prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis entailed by the premise? (yes/no/maybe)"
        prompt = format_prompt(prompt)
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        return {
            'input_ids': encoding["input_ids"],
            'attention_mask': encoding["attention_mask"],
            'labels': data["label"]
        }

    train_dataset = train_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])
    return train_dataset, eval_dataset

def get_hellaswag(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("hellaswag")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    def preprocess(data):
        context = data["ctx"]
        endings = data["endings"]
        label = int(data["label"])
        choice_text = "\n".join([f"{i + 1}. {e}" for i, e in enumerate(endings)])
        prompt = f"Context: {context}\nChoose the most plausible ending:\n{choice_text}"
        prompt = format_prompt(prompt)
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        return {
            'input_ids': encoding["input_ids"],
            'attention_mask': encoding["attention_mask"],
            'labels': label
        }

    train_dataset = train_dataset.map(preprocess, remove_columns=["ctx", "endings", "label", "activity_label", "source_id"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["ctx", "endings", "label", "activity_label", "source_id"])
    return train_dataset, eval_dataset

def get_sst2(tokenizer, args):
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    max_length = args.max_length if args.max_length > 0 else 128

    def preprocess(data):
        prompt = f"Sentiment classification:\n{data['sentence']}"
        prompt = format_prompt(prompt)
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        return {
            'input_ids': encoding["input_ids"],
            'attention_mask': encoding["attention_mask"],
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
        prompt = f"Is the following sentence grammatically correct?\n{data['sentence']}"
        prompt = format_prompt(prompt)
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        return {
            'input_ids': encoding["input_ids"],
            'attention_mask': encoding["attention_mask"],
            'labels': data["label"]
        }

    train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    return train_dataset, eval_dataset

def get_arc(tokenizer, args):
    keys = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    def preprocess(data):
        question = data['question']
        choices = list(zip(data['choices']['text'], data['choices']['label']))
        choices_text = '\n'.join([f"{label}: {text}" for text, label in choices])
        prompt = f"{question}\n{choices_text}"
        prompt = format_prompt(prompt)
        label = keys.get(data["answerKey"], -1)
        encoding = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)
        return {
            'input_ids': encoding["input_ids"],
            'attention_mask': encoding["attention_mask"],
            'labels': label
        }

    train_dataset = train_dataset.map(preprocess, remove_columns=["question", "answerKey", "choices"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["question", "answerKey", "choices"])
    train_dataset = train_dataset.filter(lambda x: x['labels'] != -1)
    eval_dataset = eval_dataset.filter(lambda x: x['labels'] != -1)
    return train_dataset, eval_dataset
