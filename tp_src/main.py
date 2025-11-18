# from comms import LossyNetwork

# #original trainer of data parallelization
# # from trainer import DistributedTrainer, MyClassifierCallback, compute_classfication_metrics

# # New trainer of distributed data parallelization
# from trainer_dist_manual import DistributedTrainer, MyClassifierCallback, compute_classfication_metrics

# from data import get_dataset
# from transformers import TrainingArguments
# import os
# import yaml
# from models import get_classifier_and_tokenizer

# def main(args):

#     with open("src/config.yaml") as config:
#         try:
#             dataset_config = yaml.safe_load(config)
#         except yaml.YAMLError as exc:
#             print(exc)
    
#     dataset_config = dataset_config[args.dataset]
#     network = LossyNetwork(loss_rate=args.loss_rate)
#     network.set_seed(args.seed)

#     model, tokenizer = get_classifier_and_tokenizer(args.model_name, num_labels=dataset_config['num_labels'])
#     train_dataset, eval_dataset = get_dataset(args, tokenizer)


#     output_dir = f"{args.output_dir}/{args.run_id}"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     with open(f"{output_dir}/args.yaml", "w") as f:
#         yaml.dump(vars(args), f)
#     args.output_dir = output_dir
#     callback_args = {
#         'report_ttac' : dataset_config['report_ttac'],
#         'report_file' : f"{args.output_dir}/ttac_report.txt",
#         'target_acc': dataset_config['target_acc'],
#     }
#     callback = MyClassifierCallback(callback_args)
    
#     # Original Data parallelization code
#     # training_args = TrainingArguments(
#     #     output_dir=output_dir,
#     #     per_device_train_batch_size=int(args.batch_size/4) if args.num_nodes == 2 else args.batch_size,
#     #     per_device_eval_batch_size=int(args.batch_size/4) if args.num_nodes == 2 else args.batch_size,
        
#     # New trainer args of distributed data parallelization
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=args.batch_size // args.num_nodes,
#         per_device_eval_batch_size=args.batch_size // args.num_nodes,
#         # From here on, it is the same
#         num_train_epochs=args.epochs,
#         learning_rate= args.learning_rate,
#         weight_decay=0.01,
#         # evaluation_strategy="steps",
#         eval_strategy="steps",
#         eval_steps=args.eval_steps,
#         save_steps=args.save_steps,
#         save_strategy="steps",
#         save_total_limit=2,
#         metric_for_best_model="accuracy",
#         logging_dir=f"{output_dir}/logs",
#         logging_steps=10,
#         fp16=args.fp16,
#         report_to="wandb"
#     )

#     trainer = DistributedTrainer(
#         num_nodes=args.num_nodes,
#         network=network,
#         model=model,
#         tokenizer=tokenizer,
#         args = training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         callbacks=[callback],
#         compute_metrics=compute_classfication_metrics,
#     )

#     trainer.train()

# # Original number of nodes was 2, I changed it to 3 for a distributed setting

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
#     parser.add_argument('--num_nodes', type=int, default=3, help='Number of nodes')
#     parser.add_argument('--loss_rate', type=float, default=0.0001, help='Packet loss rate')
#     parser.add_argument('--seed', type=int, default=1234, help='Random seed')
#     parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B', help='Model name')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
#     parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
#     parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
#     parser.add_argument('--dataset', '-d', type=str, default='winogrande', 
#                         help='Dataset to use for training')
#     parser.add_argument('--max_samples', type=int, default=0, 
#                         help='Maximum number of training samples to use (0 for all)')
#     parser.add_argument('--epochs', type=int, default=3, 
#                         help='Number of training epochs')
#     parser.add_argument('--max_length', type=int, default=256,
#                         help='Maximum sequence length for tokenization')
#     parser.add_argument('--eval_steps', type=int, default=50)
#     parser.add_argument('--save_steps', type=int, default=100)
#     parser.add_argument('--logging_steps', type=int, default=10)
#     parser.add_argument('--learning_rate', type=float, default=3e-5)
#     parser.add_argument('--run_id', type=str, required=True)
#     args = parser.parse_args()
    
#     main(args)

















# Original Pegah's code
from comms import LossyNetwork, GillbertElliotLossyNetwork
from trainer import DistributedTrainer, MyClassifierCallback, MyQACallback, MyQATrainer, compute_classfication_metrics, compute_exact_match_metric
from data import get_dataset
from transformers import TrainingArguments, Trainer
import os
import pandas as pd
import yaml
from models import get_classifier_and_tokenizer

classification_datasets = ['winogrande', 'mnli', 'sst2', 'hellaswag', 'piqa', 'arc', 'quality']
generation_datasets = ['hotpotqa', 'squad']
def main(args):

    with open("tp_src/dataset_config.yaml") as config:
        try:
            dataset_config = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = dataset_config[args.dataset]
    loss_type = args.loss_type
    if loss_type == 'ber':
        network = LossyNetwork(args)
    elif loss_type == 'g-e':
        configs = pd.read_csv('g_e_params.csv')
        original_id = args.ge_config
        model_name = args.model_name
        task = args.dataset
        seed = args.seed
        nodes = args.num_nodes
        ge_config = configs[configs['id'] == args.ge_config].iloc[0]
        network = GillbertElliotLossyNetwork(p_bg = ge_config[' pbg'],p_gb= ge_config[' pgb'],
                                             good_loss_rate=ge_config[' lrg'],
                                             bad_loss_rate=ge_config[' lrb'], args=args, loss_label=original_id,
                                             model_name=model_name, task_name=task,
                                             seed=seed, nodes=nodes)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    network.set_seed(args.seed)

    # for tasks other than classification you will need to modify the callback and the compute_metrics function, as well as get model and tokenizer
    if args.dataset in classification_datasets:
        model, tokenizer = get_classifier_and_tokenizer(args.model_name, num_labels=dataset_config['num_labels'], num_unfrozen_layers=args.num_unfrozen_layers)
        train_dataset, eval_dataset = get_dataset(args, tokenizer)
    elif args.dataset in generation_datasets:
        from models import get_qa_model_and_tokenizer
        model, tokenizer = get_qa_model_and_tokenizer(args.model_name, num_unfrozen_layers=args.num_unfrozen_layers)
        train_dataset, eval_dataset = get_dataset(args, tokenizer)

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    output_dir = f"{args.output_dir}/{args.run_id}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f) # for reproducibility

    args.output_dir = output_dir

    callback_args = { # report time to accuracy #TODO change this to "steps to accuracy"
        'report_ttac' : dataset_config['report_ttac'],
        'report_file' : f"{args.output_dir}/ttac_report.txt",
        'target_acc': dataset_config['target_acc'],
    }
    if args.dataset in generation_datasets:
        callback_args['eos_token_id'] = tokenizer.eos_token_id
        compute_metrics = compute_exact_match_metric(tokenizer)
        callback = MyQACallback(callback_args)
        trainer_class = MyQATrainer
        callback = MyQACallback(callback_args)
        trainer_class = MyQATrainer
    else:
        compute_metrics = compute_classfication_metrics
        callback = MyClassifierCallback(callback_args)
        # trainer_class = DistributedTrainer
        trainer_class = Trainer
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate= args.learning_rate,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=1,
        metric_for_best_model="accuracy" if args.dataset in classification_datasets else "exact_match",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=args.fp16,
        report_to="wandb",
        deepspeed=args.deepspeed
    )

#     trainer = trainer_class(
#         num_nodes=args.num_nodes,
#         network=network,
#         model=model,
#         tokenizer=tokenizer,
#         args = training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         callbacks=[callback],
#         compute_metrics=compute_metrics,
#     )

    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate When using Bernoulli')
    parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e'], help='Type of packet loss simulation: "ber" for Bernoulli, "g-e" for Gilbert-Elliott')
    parser.add_argument('--ge_config', type = str, default = 'default', help='configuration id for Gilbert-Elliott loss simulation. Refer to g_e_params.csv')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B', help='Model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--dataset', '-d', type=str, default='winogrande', 
                        help='Dataset to use for training')
    parser.add_argument('--max_samples', type=int, default=0, 
                        help='Maximum number of training samples to use (0 for all)')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('-nunf', '--num_unfrozen_layers', type=int, default=None, 
                        help='Number of unfrozen layers in the model. If None, all layers are unfrozen.')
    # add near the rest of your parser.add_argument(...) calls
    parser.add_argument("--local_rank", type=int, default=-1, help="set by DeepSpeed/torchrun")
    parser.add_argument("--rank", type=int, default=None, help="optional, set by launcher")
    parser.add_argument("--world_size", type=int, default=None, help="optional, set by launcher")

    parser.add_argument('--deepspeed', type=str, default=None,
                        help='Path to DeepSpeed JSON config; enables AutoTP when set.')
    args = parser.parse_args()
    
    main(args)


# My code based on Pegah's for AWS

# # main.py
# from comms import LossyNetwork, GillbertElliotLossyNetwork
# from trainer import MyClassifierCallback, MyQACallback, compute_classfication_metrics, compute_exact_match_metric, PacketLossWrapper
# from data import get_dataset
# from transformers import TrainingArguments, Trainer
# import os
# import pandas as pd
# import yaml
# from models import get_classifier_and_tokenizer

# classification_datasets = ['winogrande', 'mnli', 'sst2', 'hellaswag', 'piqa', 'arc', 'quality']
# generation_datasets = ['hotpotqa', 'squad']

# def main(args):
#     with open("src/config.yaml") as config:
#         try:
#             dataset_config = yaml.safe_load(config)
#         except yaml.YAMLError as exc:
#             print(exc)

#     dataset_config = dataset_config[args.dataset]
#     loss_type = args.loss_type
#     if loss_type == 'ber':
#         network = LossyNetwork(args)
#     elif loss_type == 'g-e':
#         configs = pd.read_csv('g_e_params.csv')
#         ge_config = configs[configs['id'] == args.ge_config].iloc[0]
#         network = GillbertElliotLossyNetwork(p_bg=ge_config[' pbg'], p_gb=ge_config[' pgb'],
#                                              good_loss_rate=ge_config[' lrg'], bad_loss_rate=ge_config[' lrb'], args=args)
#     else:
#         raise ValueError(f"Unsupported loss type: {loss_type}")
#     network.set_seed(args.seed)

#     if args.dataset in classification_datasets:
#         model, tokenizer = get_classifier_and_tokenizer(args.model_name, num_labels=dataset_config['num_labels'], num_unfrozen_layers=args.num_unfrozen_layers)
#         train_dataset, eval_dataset = get_dataset(args, tokenizer)
#     elif args.dataset in generation_datasets:
#         from models import get_qa_model_and_tokenizer
#         model, tokenizer = get_qa_model_and_tokenizer(args.model_name, num_unfrozen_layers=args.num_unfrozen_layers)
#         train_dataset, eval_dataset = get_dataset(args, tokenizer)
#     else:
#         raise ValueError(f"Unsupported dataset: {args.dataset}")

#     # Wrap the model with packet loss simulation
#     model = PacketLossWrapper(model, network)

#     output_dir = f"{args.output_dir}/{args.run_id}"
#     os.makedirs(output_dir, exist_ok=True)

#     with open(f"{output_dir}/args.yaml", "w") as f:
#         yaml.dump(vars(args), f)

#     args.output_dir = output_dir

#     callback_args = {
#         'report_ttac': dataset_config['report_ttac'],
#         'report_file': f"{args.output_dir}/ttac_report.txt",
#         'target_acc': dataset_config['target_acc'],
#     }

#     if args.dataset in generation_datasets:
#         callback_args['eos_token_id'] = tokenizer.eos_token_id
#         compute_metrics = compute_exact_match_metric(tokenizer)
#         callback = MyQACallback(callback_args)
#     else:
#         compute_metrics = compute_classfication_metrics
#         callback = MyClassifierCallback(callback_args)

#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         num_train_epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         remove_unused_columns=False,
#         optim="adamw_bnb_8bit",
#         weight_decay=0.01,
#         eval_strategy="steps",
#         eval_steps=args.eval_steps,
#         save_steps=args.save_steps,
#         save_strategy="steps",
#         save_total_limit=1,
#         metric_for_best_model="accuracy" if args.dataset in classification_datasets else "exact_match",
#         logging_dir=f"{output_dir}/logs",
#         logging_steps=10,
#         # fp16=args.fp16,
#         fp16=True,
#         report_to="wandb"
#     )

#     trainer = Trainer(
#         model=model,
#         tokenizer=tokenizer,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         callbacks=[callback],
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
#     parser.add_argument('--num_nodes', type=int, default=2)
#     parser.add_argument('--loss_rate', type=float, default=0.001)
#     parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e'])
#     parser.add_argument('--ge_config', type=str, default='default')
#     parser.add_argument('--seed', type=int, default=1234)
#     parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--fp16', action='store_true')
#     parser.add_argument('--output_dir', type=str, default='./output')
#     parser.add_argument('--dataset', '-d', type=str, default='winogrande')
#     parser.add_argument('--max_samples', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=3)
#     parser.add_argument('--max_length', type=int, default=256)
#     parser.add_argument('--eval_steps', type=int, default=50)
#     parser.add_argument('--save_steps', type=int, default=100)
#     parser.add_argument('--logging_steps', type=int, default=10)
#     parser.add_argument('--learning_rate', type=float, default=3e-5)
#     parser.add_argument('--run_id', type=str, required=True)
#     parser.add_argument('--num_unfrozen_layers', type=int, default=None)
#     args = parser.parse_args()

#     main(args)
