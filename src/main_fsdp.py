import os
import yaml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from comms import LossyNetwork
from transformers import Trainer, TrainingArguments
from trainer import MyClassifierCallback, compute_classfication_metrics
from data import get_dataset
from models import get_classifier_and_tokenizer

def main(args):
    # 1) Initialize distributed and bind GPU
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 2) Load dataset config
    with open("src/config.yaml") as config_file:
        all_cfg = yaml.safe_load(config_file)
    dataset_cfg = all_cfg[args.dataset]

    # 3) Instantiate your LossyNetwork (no changes here)
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    # 4) Load model/tokenizer, move to device
    model, tokenizer = get_classifier_and_tokenizer(
        args.model_name,
        num_labels=dataset_cfg["num_labels"],
    )
    model.to(device)

    # 5) Wrap model in FSDP for tensor parallelism
    model = FSDP(
        model,
        mixed_precision=torch.float16 if args.fp16 else torch.float32,
        backward_prefetch="backward_pre",
    )

    # 6) Register backward hooks on every parameter to inject packet loss
    def make_lossy_hook(net):
        def hook(grad):
            mask = net.send(grad)
            return net.receive(grad, mask)
        return hook

    for _, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_lossy_hook(network))

    # 7) Prepare datasets
    train_dataset, eval_dataset = get_dataset(args, tokenizer)
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    eval_sampler  = DistributedSampler(eval_dataset, shuffle=False)

    # 8) Save args & create output dir
    output_dir = f"{args.output_dir}/{args.run_id}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f)
    args.output_dir = output_dir

    # 9) Set up callbacks & TrainingArguments
    callback_args = {
        "report_ttac": dataset_cfg["report_ttac"],
        "report_file": f"{output_dir}/ttac_report.txt",
        "target_acc":  dataset_cfg["target_acc"],
    }
    callback = MyClassifierCallback(callback_args)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(args.batch_size / args.num_nodes),
        per_device_eval_batch_size=int(args.batch_size / args.num_nodes),
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
        report_to="wandb",
    )

    # 10) Initialize HF Trainer and train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback],
        compute_metrics=compute_classfication_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes',    type=int,   default=2,           help='Number of nodes')
    parser.add_argument('--loss_rate',    type=float, default=0.001,       help='Packet loss rate')
    parser.add_argument('--seed',         type=int,   default=1234,        help='Random seed')
    parser.add_argument('--model_name',   type=str,   default='meta-llama/Llama-3.2-1B', help='Model name')
    parser.add_argument('--batch_size',   type=int,   default=64,          help='Batch size')
    parser.add_argument('--fp16',         action='store_true',             help='Use mixed precision')
    parser.add_argument('--output_dir',   type=str,   default='./output',   help='Output directory')
    parser.add_argument('--dataset', '-d', type=str,   default='winogrande',help='Dataset to use')
    parser.add_argument('--max_samples',  type=int,   default=0,           help='Max samples (0=all)')
    parser.add_argument('--epochs',       type=int,   default=3,           help='Num epochs')
    parser.add_argument('--max_length',   type=int,   default=256,         help='Token max length')
    parser.add_argument('--eval_steps',   type=int,   default=50,          help='Eval every N steps')
    parser.add_argument('--save_steps',   type=int,   default=100,         help='Save every N steps')
    parser.add_argument('--logging_steps',type=int,   default=10,          help='Log every N steps')
    parser.add_argument('--learning_rate',type=float, default=3e-5,        help='Learning rate')
    parser.add_argument('--run_id',       type=str,   required=True,      help='Run identifier')
    args = parser.parse_args()
    main(args)
