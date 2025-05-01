from comms import LossyNetwork

#original trainer of data parallelization
# from trainer import DistributedTrainer, MyClassifierCallback, compute_classfication_metrics

# New trainer of distributed data parallelization
from trainer_dist_manual import DistributedTrainer, MyClassifierCallback, compute_classfication_metrics

from data import get_dataset
from transformers import TrainingArguments
import os
import yaml
from models import get_classifier_and_tokenizer

def main(args):

    with open("src/config.yaml") as config:
        try:
            dataset_config = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = dataset_config[args.dataset]
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    model, tokenizer = get_classifier_and_tokenizer(args.model_name, num_labels=dataset_config['num_labels'])
    train_dataset, eval_dataset = get_dataset(args, tokenizer)


    output_dir = f"{args.output_dir}/{args.run_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    args.output_dir = output_dir
    callback_args = {
        'report_ttac' : dataset_config['report_ttac'],
        'report_file' : f"{args.output_dir}/ttac_report.txt",
        'target_acc': dataset_config['target_acc'],
    }
    callback = MyClassifierCallback(callback_args)
    
    # Original Data parallelization code
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=int(args.batch_size/4) if args.num_nodes == 2 else args.batch_size,
    #     per_device_eval_batch_size=int(args.batch_size/4) if args.num_nodes == 2 else args.batch_size,
        
    # New trainer args of distributed data parallelization
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size // args.num_nodes,
        per_device_eval_batch_size=args.batch_size // args.num_nodes,
        # From here on, it is the same
        num_train_epochs=args.epochs,
        learning_rate= args.learning_rate,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        fp16=args.fp16,
        report_to="wandb"
    )

    trainer = DistributedTrainer(
        num_nodes=args.num_nodes,
        network=network,
        model=model,
        tokenizer=tokenizer,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
        compute_metrics=compute_classfication_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate')
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
    args = parser.parse_args()
    
    main(args)
