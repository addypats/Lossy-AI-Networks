# from datasets import load_dataset

# def get_dataset(args, tokenizer):
#     if args.dataset == "sst2":
#         return get_sst2(tokenizer, args)
#     elif args.dataset == "cola":
#         return get_cola(tokenizer, args)
#     elif args.dataset == "mnli":
#         return get_mnli(tokenizer, args)
#     elif args.dataset == "winogrande":
#         return get_winogrande(tokenizer, args)
#     elif args.dataset == "arc":
#         return get_arc(tokenizer, args)
#     elif args.dataset == "hellaswag":
#         return get_hellaswag(tokenizer, args)
#     elif args.dataset == "piqa": 
#         return get_piqa(tokenizer, args)
#     else:
#         raise ValueError(f"Dataset {args.dataset} not supported.")


# def get_winogrande(tokenizer, args):

#     max_length = args.max_length if args.max_length > 0 else 256
#     dataset = load_dataset("allenai/winogrande", "winogrande_l", trust_remote_code=True)    
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["validation"]

#     def preprocess(data):
#         # Improve Winogrande preprocessing by formatting the input as a choice task
#         # Replace the placeholder '_' with each option and tokenize separately
#         sentence = data["sentence"]
#         option1 = data["option1"]
#         option2 = data["option2"]
        
#         # Find the placeholder marker in the sentence
#         if "_" in sentence:
#             placeholder = "_"
#         else:
#             # Some versions might use a different placeholder
#             placeholder = "___"
            
#         # Create two complete sentences with each option
#         sentence1 = sentence.replace(placeholder, option1)
#         sentence2 = sentence.replace(placeholder, option2)
        
#         # Use the correct option as the answer
#         label = 0 if data["answer"] == '1' else 1
        
#         # Tokenize for binary classification - encode each option separately
#         encodings = tokenizer(
#             [sentence1, sentence2],
#             truncation=True,
#             padding="max_length", 
#             max_length=max_length,
#             return_tensors="pt"
#         )
        
#         # Select the encoding of the correct option
#         return {
#             'input_ids': encodings["input_ids"][label].tolist(),
#             'attention_mask': encodings["attention_mask"][label].tolist(),
#             'labels': label
#         }
#     train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])
#     eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "option1", "option2", "answer"])

#     return train_dataset, eval_dataset

# def get_sst2(tokenizer, args):

#     dataset = load_dataset("glue", "sst2")
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["validation"]

#     max_length = args.max_length if args.max_length > 0 else 128

#     def preprocess(data):
#         return {
#             'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
#             'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
#             'labels': data["label"]
#         }
        
#     train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
#     eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
    
#     return train_dataset, eval_dataset


# def get_cola(tokenizer, args):
#     max_length = args.max_length if args.max_length > 0 else 128
#     dataset = load_dataset("glue", "cola")
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["validation"]

#     def preprocess(data):
#         return {
#             'input_ids': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
#             'attention_mask': tokenizer(data["sentence"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
#             'labels': data["label"]
#         }
        
#     train_dataset = train_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])
#     eval_dataset = eval_dataset.map(preprocess, remove_columns=["sentence", "idx", "label"])

#     return train_dataset, eval_dataset

# def get_mnli(tokenizer, args):
#     max_length = args.max_length if args.max_length > 0 else 128
#     dataset = load_dataset("glue", "mnli")
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["validation_matched"]  # Using matched validation set

#     def preprocess(data):
#         return {
#             'input_ids': tokenizer(data["premise"], data["hypothesis"], truncation=True, padding="max_length", max_length=max_length)["input_ids"],
#             'attention_mask': tokenizer(data["premise"], data["hypothesis"], truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
#             'labels': data["label"]
#         }
        
#     train_dataset = train_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])
#     eval_dataset = eval_dataset.map(preprocess, remove_columns=["premise", "hypothesis", "idx", "label"])

#     return train_dataset, eval_dataset

# def get_arc(tokenizer, args):

#     keys = {
#         'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3,
#     }
#     max_length = args.max_length if args.max_length > 0 else 256
#     dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["validation"]
#     def preprocess(data):

#         question = data['question']
#         choices = list(zip(data['choices']['text'], data['choices']['label']))
        
#         choices = ' '.join([f"{label}: {text}" for text, label in choices])
#         question = f"{question}\n\n{choices}"
#         return {
#             'input_ids': tokenizer(question, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
#             'attention_mask': tokenizer(question, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
#             'labels': keys.get(data["answerKey"], -1)
#         }
        
#     train_dataset = train_dataset.map(preprocess, remove_columns=["question", "answerKey", "choices"])
#     eval_dataset = eval_dataset.map(preprocess, remove_columns=["question", "answerKey", "choices"])

#     # Filter out invalid labels
#     train_dataset = train_dataset.filter(lambda x: x['labels'] != -1)
#     eval_dataset = eval_dataset.filter(lambda x: x['labels'] != -1)

#     return train_dataset, eval_dataset

# def get_hellaswag(tokenizer, args):
#     max_length = args.max_length if args.max_length > 0 else 256
#     dataset = load_dataset("hellaswag")
    
#     # Limit dataset size if specified
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["validation"]
    
#     if args.max_samples > 0:
#         train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    
#     def preprocess(data):
#         # Format context and endings for multiple choice
#         context = data["ctx"]
#         endings = data["endings"]
        
#         # Format as multiple choice
#         choices = [f"{context} {ending}" for ending in endings]
        
#         # Create encodings for all choices
#         encodings = tokenizer(
#             choices,
#             truncation=True,
#             padding="max_length",
#             max_length=max_length,
#             return_tensors="pt"
#         )
        
#         # Get label (correct ending index)
#         label = int(data["label"])
        
#         return {
#             'input_ids': encodings["input_ids"][label].tolist(),
#             'attention_mask': encodings["attention_mask"][label].tolist(),
#             'labels': label
#         }
    
#     # Map preprocessing function to datasets
#     train_dataset = train_dataset.map(preprocess, remove_columns=["ctx", "endings", "label", "activity_label", "source_id"])
#     eval_dataset = eval_dataset.map(preprocess, remove_columns=["ctx", "endings", "label", "activity_label", "source_id"])
    
#     return train_dataset, eval_dataset

# def get_piqa(tokenizer, args):
#     max_length = args.max_length if args.max_length > 0 else 256
#     dataset = load_dataset("piqa", trust_remote_code=True)
    
#     def preprocess(data):
        
#         goal = data["goal"]
#         sol1 = data['sol1']
#         sol2 = data['sol2']
                
#         input = "Which of the following is a better solution to the problem?\n\n" + f" 1) {goal} {sol1} \n 2) {goal} {sol2}"
#         label = data["label"]

        
#         return {
#             'input_ids': tokenizer(input, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
#             'attention_mask': tokenizer(input, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
#             'labels': label
#         }
    
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["validation"]

#     train_dataset = train_dataset.map(preprocess, remove_columns=["goal", "sol1", "sol2", "label"])
#     eval_dataset = eval_dataset.map(preprocess, remove_columns=["goal", "sol1", "sol2", "label"])
    
#     return train_dataset, eval_dataset



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
    elif args.dataset == "quality":
        return get_quality(tokenizer, args)
    elif args.dataset == "hotpotqa":
        return get_hotpotqa(tokenizer, args)
    elif args.dataset == "squad":
        return get_squad(tokenizer, args)
    elif args.dataset == "tinysquad":
        return get_tinysquad(tokenizer, args)
    elif args.dataset == "newsqa":
        return get_newsqa(tokenizer, args)
    elif args.dataset == "triviaqa":
        return get_triviaqa(tokenizer, args)
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


def get_cosmosqa(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 512
    dataset = load_dataset("allenai/cosmos_qa")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    def preprocess(data):
        context = data['context']
        question = data['question']
        answers = f"""
        1: {data['answer0']}\n
        2: {data['answer1']}\n
        3: {data['answer2']}\n
        4: {data['answer3']}\n
        """
        question = f"{question}\n\n{answers}"
        all_text = f"{context}\n\n{question}"
        return {
            'input_ids': tokenizer(all_text, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(all_text, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': data['label']
        }
    train_dataset = train_dataset.map(preprocess, remove_columns=["id","context", "question", "answer0", "answer1", "answer2", "answer3", "label"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["id","context", "question", "answer0", "answer1", "answer2", "answer3", "label"])

    return train_dataset, eval_dataset

def get_arc(tokenizer, args):

    keys = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3,
    }
    choice_unification = {
        'A': '1', 'B': '2', 'C': '3', 'D': '4',
        '1': '1', '2': '2', '3': '3', '4': '4'
    }
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    def preprocess(data):
        question = data['question']
        # normalize each label individually
        normalized_labels = [choice_unification.get(lbl, lbl) for lbl in data['choices']['label']]
        choices = list(zip(data['choices']['text'], normalized_labels))

        # format choices into the prompt
        choices_str = '\n'.join([f"{label}: {text}" for text, label in choices])
        prompt = f"{question}\n\n{choices_str}"

        tokenized = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        return {
            'input_ids': tokenized["input_ids"],
            'attention_mask': tokenized["attention_mask"],
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
        
        
        prefix = "Which of the following is the most likely continuation of the context?\n\n"
        question = f"{prefix}{context}\n\n Choices:\n"
        choices = [f"{i+1}: {ending}" for i, ending in enumerate(endings)]
        question += "\n".join(choices)

        question_encoded = tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return {
            'input_ids': question_encoded["input_ids"],
            'attention_mask': question_encoded["attention_mask"],
            'labels': int(data["label"])
        }
    
    # Map preprocessing function to datasets
    train_dataset = train_dataset.map(preprocess, remove_columns=["ctx_a", "ctx_b","ctx", "endings", "label", "activity_label", "source_id"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["ctx_a", "ctx_b","ctx", "endings", "label", "activity_label", "source_id"])
    
    return train_dataset, eval_dataset


def get_piqa(tokenizer, args):
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("piqa")
    
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


def get_quality(tokenizer, args):
    max_length = 10000
    dataset = load_dataset("emozilla/quality", trust_remote_code=True)
    def preprocess(data):
        question = data["question"]
        article = data["article"]
        options = data["options"]
        answer = data["answer"]

        input_text = f""" Answer the question based on the article below. Choose the best answer from the options provided.\n
        Article: {article}\n
        Question: {question}\n
        Options: 1. {options[0]}\n 2. {options[1]}\n 3. {options[2]}\n 4. {options[3]}\n
        """
        answer = int(answer)

        return {
            'input_ids': tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
            'attention_mask': tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
            'labels': answer
        }
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    # random sample eval dataset 
    eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(min(800, len(eval_dataset))))
    train_dataset = train_dataset.map(preprocess, remove_columns=["question", "article", "options", "answer", "hard"])
    eval_dataset = eval_dataset.map(preprocess, remove_columns=["question", "article", "options", "answer", "hard"])
    return train_dataset, eval_dataset


# def get_hotpotqa(tokenizer, args):
#     max_length = 1000
#     dataset = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
#     def preprocess(data): 
#         context = ""
#         for i, title in enumerate(data["context"]["title"]):
#             context += f"{title}:\n {''.join(data['context']['sentences'][i])}"
#         input_text = f"""
#         Answer the question based on the context. The context is a collection of Wikipedia articles. Answer with a single word or phrase.\n
#         ## Context:\n{context}\n
#         ## Question: {data['question']}\n
#         ## Answer: {data['answer']}\n
#         """

#         d = {'input_ids': tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
#              'attention_mask': tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
#              'labels': tokenizer(data['answer'], truncation=True, padding="max_length", max_length=max_length)["input_ids"]}
#         # Remove padding tokens from labels
#         d['labels'] = [label if label != tokenizer.pad_token_id else -100 for label in d['labels']]
#         return d
#     if args.max_samples > 0:
#         train_dataset = dataset['train'].shuffle(seed=args.seed).select(range(min(args.max_samples, len(dataset["train"]))))
#         eval_dataset = dataset['validation'].shuffle(seed=args.seed).select(range(min(32, len(dataset["validation"]))))
#     else:
#         train_dataset = dataset["train"]
#         eval_dataset = dataset["validation"]
#     train_dataset = train_dataset.map(preprocess)
#     eval_dataset = eval_dataset.map(preprocess)
#     return train_dataset, eval_dataset

from datasets import load_dataset

def get_hotpotqa(tokenizer, args):
    max_length = args.max_length if hasattr(args, "max_length") else 256
    dataset = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)

    def preprocess(example):
        # Build context string
        context = ""
        for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
            context += f"{title}:\n{''.join(sentences)}\n"

        # Prompt does NOT include the gold answer
        prompt = (
            "Answer the question based on the context. The context is a collection of Wikipedia articles. "
            "Answer with a single word or phrase.\n\n"
            "## Context:\n"
            f"{context}\n"
            f"## Question: {example['question']}\n"
            "## Answer:"
        )

        # Tokenize prompt and gold answer separately
        inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        label_enc = tokenizer(
            example["answer"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        labels = label_enc["input_ids"]
        # Replace pad token id with -100 so loss ignores padding
        labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    if args.max_samples > 0:
        train_dataset = dataset["train"].shuffle(seed=args.seed).select(
            range(min(args.max_samples, len(dataset["train"])))
        )
        eval_dataset = dataset["validation"].shuffle(seed=args.seed).select(
            range(min(32, len(dataset["validation"])))
        )
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

    train_dataset = train_dataset.map(preprocess)
    eval_dataset = eval_dataset.map(preprocess)
    return train_dataset, eval_dataset


# Original working implementation of squad - use this if the other does not work

# def get_squad(tokenizer, args):
#     max_length = args.max_length if args.max_length > 0 else 512
#     dataset = load_dataset("squad")

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         model.resize_token_embeddings(len(tokenizer))  # if you have access here
#         model.config.pad_token_id = tokenizer.pad_token_id

#     def preprocess(example):
#         context = example["context"]
#         question = example["question"]
#         answer = example["answers"]["text"][0] if example["answers"]["text"] else ""

#         prompt = (
#             "You are a helpful question-answering assistant.\n\n"
#             f"Context:\n{context}\n\n"
#             f"Question:\n{question}\n\n"
#             "Answer:"
#         )

#         full_text = prompt + " " + answer + tokenizer.eos_token

#         tokenized = tokenizer(
#             full_text,
#             truncation=True,
#             max_length=max_length,
#             padding="max_length"
#         )

#         input_ids = tokenized["input_ids"]

#         # Mask prompt tokens in labels so loss is only on the answer
#         prompt_ids = tokenizer(
#             prompt,
#             truncation=True,
#             max_length=max_length,
#             add_special_tokens=False
#         )["input_ids"]

#         labels = input_ids.copy()
#         labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
#         tokenized["labels"] = labels

#         return tokenized

#     train_dataset = dataset["train"].map(
#         preprocess,
#         remove_columns=dataset["train"].column_names
#     )
#     eval_dataset = dataset["validation"].map(
#         preprocess,
#         remove_columns=dataset["validation"].column_names
#     )

#     return train_dataset, eval_dataset


def _build_qa_example(tokenizer, context, question, answer, max_length):
    """
    Shared helper to create a generative QA example:
    prompt = system + context + question, label = answer (prompt tokens masked with -100).
    """
    if answer is None:
        answer = ""
    answer = str(answer)

    prompt = (
        "You are a helpful question-answering assistant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    full_text = prompt + " " + answer + (tokenizer.eos_token or "")

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    input_ids = tokenized["input_ids"]

    # Mask prompt tokens in labels so loss is only on the answer
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )["input_ids"]

    labels = input_ids.copy()
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)
    tokenized["labels"] = labels

    return tokenized


def get_squad(tokenizer, args):
    # Full SQuAD – this is the heavy one
    max_length = args.max_length if args.max_length > 0 else 512
    dataset = load_dataset("squad")

    def preprocess(example):
        context = example["context"]
        question = example["question"]
        # SQuAD has answers["text"] list
        answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
        return _build_qa_example(tokenizer, context, question, answer, max_length)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    # Optional subsampling for memory-scarce setups
    if getattr(args, "max_samples", 0) > 0:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(
            range(min(args.max_samples, len(train_dataset)))
        )

    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        preprocess,
        remove_columns=eval_dataset.column_names,
    )
    return train_dataset, eval_dataset


def get_tinysquad(tokenizer, args):
    """
    TinySQuAD (zakerytclarke/tinysquad):
    - Single 'train' split with fields: title, context, question, answer
    - We'll create a small validation split ourselves.
    """
    max_length = args.max_length if args.max_length > 0 else 256
    raw = load_dataset("zakerytclarke/tinysquad")  # single 'train' split

    full_train = raw["train"]

    # Optional: limit training size to avoid OOM
    if getattr(args, "max_samples", 0) > 0:
        full_train = full_train.shuffle(seed=args.seed).select(
            range(min(args.max_samples, len(full_train)))
        )

    # 90/10 train/val split from the single split
    split = full_train.train_test_split(test_size=0.1, seed=args.seed)
    train_raw = split["train"]
    eval_raw = split["test"]

    def preprocess(example):
        context = example["context"]
        question = example["question"]
        answer = example["answer"]
        return _build_qa_example(tokenizer, context, question, answer, max_length)

    train_dataset = train_raw.map(
        preprocess,
        remove_columns=train_raw.column_names,
    )
    eval_dataset = eval_raw.map(
        preprocess,
        remove_columns=eval_raw.column_names,
    )
    return train_dataset, eval_dataset


def get_newsqa(tokenizer, args):
    """
    NewsQA (lucadiliello/newsqa):
    - Splits: train, validation
    - Fields: context, question, labels (list of answer strings)
    """
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("lucadiliello/newsqa")

    train_raw = dataset["train"]
    eval_raw = dataset["validation"]

    if getattr(args, "max_samples", 0) > 0:
        train_raw = train_raw.shuffle(seed=args.seed).select(
            range(min(args.max_samples, len(train_raw)))
        )

    def preprocess(example):
        context = example["context"]
        question = example["question"]
        labels = example.get("labels", [])
        answer = labels[0] if labels else ""
        return _build_qa_example(tokenizer, context, question, answer, max_length)

    train_dataset = train_raw.map(
        preprocess,
        remove_columns=train_raw.column_names,
    )
    eval_dataset = eval_raw.map(
        preprocess,
        remove_columns=eval_raw.column_names,
    )
    return train_dataset, eval_dataset


def get_triviaqa(tokenizer, args):
    """
    TriviaQA (lucadiliello/triviaqa):
    - Splits: train, validation
    - Fields: context, question, labels (list of answer strings)
    """
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("lucadiliello/triviaqa")

    train_raw = dataset["train"]
    eval_raw = dataset["validation"]

    if getattr(args, "max_samples", 0) > 0:
        train_raw = train_raw.shuffle(seed=args.seed).select(
            range(min(args.max_samples, len(train_raw)))
        )

    def preprocess(example):
        context = example["context"]
        question = example["question"]
        labels = example.get("labels", [])
        answer = labels[0] if labels else ""
        return _build_qa_example(tokenizer, context, question, answer, max_length)

    train_dataset = train_raw.map(
        preprocess,
        remove_columns=train_raw.column_names,
    )
    eval_dataset = eval_raw.map(
        preprocess,
        remove_columns=eval_raw.column_names,
    )
    return train_dataset, eval_dataset

