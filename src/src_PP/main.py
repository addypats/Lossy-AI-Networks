# main.py
from comms import LossyNetwork
from trainer_pipe import DistributedTrainerWithPipe, MyClassifierCallback, compute_classfication_metrics
from data import get_dataset
from transformers import TrainingArguments
import os
import yaml
import argparse
from models import get_classifier_and_tokenizer
import torch.distributed.rpc as rpc

def main(args):

    with open("src/src_PP/dataset_config.yaml") as config:
        try:
            dataset_config = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = dataset_config[args.dataset]
    
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    #
    #    Give each worker a unique name (e.g. "worker0", "worker1", …).
    #
    worker_name = f"worker{rank}"

    # Choose an RPC backend; commonly "tensorpipe" (for GPU) or "gloo".  
    # Below we use TensorPipe with the default init_method="env://".
    #
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(init_method="env://")
    rpc.init_rpc(
        name=worker_name,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )
    
    network = LossyNetwork(args)
    network.set_seed(args.seed)

    model, tokenizer = get_classifier_and_tokenizer(args.model_name, num_labels=dataset_config['num_labels'], num_unfrozen_layers=args.num_unfrozen_layers)
    train_dataset, eval_dataset = get_dataset(args, tokenizer)

    output_dir = f"{args.output_dir}/{args.run_id}"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    args.output_dir = output_dir

    callback_args = {
        'report_ttac': dataset_config['report_ttac'],
        'report_file': f"{args.output_dir}/ttac_report.txt",
        'target_acc': dataset_config['target_acc'],
    }

    callback = MyClassifierCallback(callback_args)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        report_to="wandb"
    )

    trainer = DistributedTrainerWithPipe(
        num_nodes=args.num_nodes,
        network=network,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
        compute_metrics=compute_classfication_metrics,
    )

    trainer.train()
    
    rpc.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--loss_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--dataset', '-d', type=str, default='winogrande')
    parser.add_argument('--max_samples', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('-nunf', '--num_unfrozen_layers', type=int, default=None)
    args = parser.parse_args()

    main(args)
