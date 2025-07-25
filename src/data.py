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
    max_length = args.max_length if args.max_length > 0 else 256
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    def preprocess(data):

        question = data['question']
        choices = list(zip(data['choices']['text'], data['choices']['label']))
        
        choices = '\n'.join([f"{label}: {text}" for text, label in choices])
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


def get_hotpotqa(tokenizer, args):
    max_length = 1000
    dataset = load_dataset("hotpot_qa", "distractor", trust_remote_code=True)
    def preprocess(data): 
        context = ""
        for i, title in enumerate(data["context"]["title"]):
            context += f"{title}:\n {''.join(data['context']['sentences'][i])}"
        input_text = f"""
        Answer the question based on the context. The context is a collection of Wikipedia articles. Answer with a single word or phrase.\n
        ## Context:\n{context}\n
        ## Question: {data['question']}\n
        ## Answer: {data['answer']}\n
        """

        d = {'input_ids': tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)["input_ids"],
             'attention_mask': tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length)["attention_mask"],
             'labels': tokenizer(data['answer'], truncation=True, padding="max_length", max_length=max_length)["input_ids"]}
        # Remove padding tokens from labels
        d['labels'] = [label if label != tokenizer.pad_token_id else -100 for label in d['labels']]
        return d
    if args.max_samples > 0:
        train_dataset = dataset['train'].shuffle(seed=args.seed).select(range(min(args.max_samples, len(dataset["train"]))))
        eval_dataset = dataset['validation'].shuffle(seed=args.seed).select(range(min(32, len(dataset["validation"]))))
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    train_dataset = train_dataset.map(preprocess)
    eval_dataset = eval_dataset.map(preprocess)
    return train_dataset, eval_dataset

def get_squad(tokenizer, args):
    max_length = 1024
    # dataset = load_dataset("rajpurkar/squad")
    dataset = load_dataset("squad")
    
    # def preprocess(data, is_train=True):
    #     context = data["context"]
    #     question = data["question"]
    #     answer = data["answers"]["text"][0] if data["answers"]["text"] else ""
        
    #     # Format as reading comprehension task (following hotpotqa pattern)
    #     input_text = f"""Answer the question based on the context provided. Provide a concise answer extracted from the context.

    #     Context: {context}
        
    #     Question: {question}

    #     Answer:"""

    #     answer = answer +' ' + tokenizer.eos_token

    #     if is_train:
    #         input_text += f" {answer}"
    #         input_encoding = tokenizer(
    #             input_text,
    #             truncation=True,
    #             padding="max_length",
    #             max_length=max_length,
    #             add_special_tokens=True,
    #         )
        
    #         prompt_len = sum(
    #         1 for id in input_encoding["input_ids"] if id != tokenizer.pad_token_id
    #         )

    #         labels = input_encoding["input_ids"].copy()
    #         labels[:prompt_len] = [-100] * prompt_len
    #         labels = input_encoding["input_ids"].copy()
    #         labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        
    #     return {
    #         'input_ids': input_encoding["input_ids"],
    #         'attention_mask': input_encoding["attention_mask"],
    #         'labels': labels
    #     }
    

    def preprocess(data, is_train=True):
        context = data["context"]
        question = data["question"]
        answer = data["answers"]["text"][0] if data["answers"]["text"] else ""
        
        # Format as reading comprehension task
        input_text = f"""Answer the question based on the context provided. Provide a concise answer extracted from the context.

Context: {context}

Question: {question}
Answer:"""
        
        if is_train:
            # For training, include the answer
            full_text = input_text + f" {answer}" + tokenizer.eos_token
            input_encoding = tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                add_special_tokens=True,
            )
            
            # Only compute loss on answer tokens
            prompt_encoding = tokenizer(
                input_text,
                truncation=True,
                padding=False,
                add_special_tokens=True,
            )
            
            # Create labels: -100 for prompt tokens, actual tokens for answer
            labels = input_encoding["input_ids"].copy()
            prompt_length = len(prompt_encoding["input_ids"])
            
            # Mask out the prompt tokens
            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100
                
            # Mask out padding tokens  
            labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
            
            return {
                'input_ids': input_encoding["input_ids"],
                'attention_mask': input_encoding["attention_mask"],
                'labels': labels
            }
        else:
            # For evaluation, don't include the answer in input
            input_encoding = tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                add_special_tokens=True,
            )
            
            # For evaluation, labels should be just the answer for metric computation
            answer_encoding = tokenizer(
                f"Answer: {answer}",
                truncation=True,
                padding="max_length",
                max_length=100,  # Shorter length for answers
                add_special_tokens=False,
            )
            
            return {
                'input_ids': input_encoding["input_ids"],
                'attention_mask': input_encoding["attention_mask"],
                'labels': answer_encoding["input_ids"]
            }
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    if args.max_samples > 0:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(
            range(min(args.max_samples, len(train_dataset)))
        )
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(
            range(min(args.max_samples // 10, len(eval_dataset)))
        )
    # Apply preprocessing
    train_dataset = train_dataset.map(
        preprocess, 
        remove_columns=["id", "title", "context", "question", "answers"],
        fn_kwargs={"is_train": True}
    )
    eval_dataset = eval_dataset.map(
        preprocess, 
        remove_columns=["id", "title", "context", "question", "answers"],
        fn_kwargs={"is_train": False}
    )
    
    
    return train_dataset, eval_dataset
