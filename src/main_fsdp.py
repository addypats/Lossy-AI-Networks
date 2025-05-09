import os
import yaml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision
)

from comms import LossyNetwork
from transformers import Trainer, TrainingArguments
from trainer import MyClassifierCallback, compute_classfication_metrics
from data import get_dataset
from models import get_classifier_and_tokenizer

def main(args):
    # 1) DDP init & device binding
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 2) Load dataset config
    with open("src/config.yaml") as f:
        cfg = yaml.safe_load(f)
    dataset_cfg = cfg[args.dataset]

    # 3) LossyNetwork (unchanged)
    network = LossyNetwork(loss_rate=args.loss_rate)
    network.set_seed(args.seed)

    # 4) Load & move model
    model, tokenizer = get_classifier_and_tokenizer(
        args.model_name,
        num_labels=dataset_cfg["num_labels"],
    )
    model.to(device)

    # 5) Build a MixedPrecision object
    if args.fp16:
        mp = MixedPrecision(
            param_dtype=torch.half,
            reduce_dtype=torch.half,
            buffer_dtype=torch.half
        )
    else:
        mp = MixedPrecision(
            param_dtype=torch.float,
            reduce_dtype=torch.float,
            buffer_dtype=torch.float
        )

    # 6) Wrap in FSDP
    model = FSDP(
        model,
        mixed_precision=mp,
        backward_prefetch="backward_pre",
    )

    # 7) Hook in packet-loss on every gradient
    def make_lossy_hook(net):
        def hook(grad):
            mask = net.send(grad)
            return net.receive(grad, mask)
        return hook

    for _, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(make_lossy_hook(network))

    # 8) Prepare datasets & samplers
    train_ds, eval_ds = get_dataset(args, tokenizer)
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(train_ds)
    eval_sampler  = DistributedSampler(eval_ds, shuffle=False)

    # 9) Save args + create output dir
    output_dir = f"{args.output_dir}/{args.run_id}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/args.yaml", "w") as out:
        yaml.dump(vars(args), out)

    # 10) Callbacks & TrainingArguments
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
        remove_unused_columns=False,
    )

    # 11) Run HF Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[callback],
        compute_metrics=compute_classfication_metrics,
    )
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FSDP + LossyNetwork Training")
    parser.add_argument('--num_nodes',    type=int,   default=2,          help='Processes/GPUs per node')
    parser.add_argument('--loss_rate',    type=float, default=0.001,      help='Packet loss rate')
    parser.add_argument('--seed',         type=int,   default=1234,       help='Random seed')
    parser.add_argument('--model_name',   type=str,   default='meta-llama/Llama-3.2-1B', help='HF model')
    parser.add_argument('--batch_size',   type=int,   default=64,         help='Global batch size')
    parser.add_argument('--fp16',         action='store_true',            help='Enable mixed precision')
    parser.add_argument('--output_dir',   type=str,   default='./output',  help='Output dir')
    parser.add_argument('--dataset', '-d', type=str,   default='winogrande', help='Dataset key')
    parser.add_argument('--max_samples',  type=int,   default=0,          help='Max samples (0=all)')
    parser.add_argument('--epochs',       type=int,   default=3,          help='Number of epochs')
    parser.add_argument('--max_length',   type=int,   default=256,        help='Token max length')
    parser.add_argument('--eval_steps',   type=int,   default=50,         help='Eval every N steps')
    parser.add_argument('--save_steps',   type=int,   default=100,        help='Save every N steps')
    parser.add_argument('--logging_steps',type=int,   default=10,         help='Log every N steps')
    parser.add_argument('--learning_rate',type=float, default=3e-5,       help='Learning rate')
    parser.add_argument('--run_id',       type=str,   required=True,     help='Unique run ID')
    args = parser.parse_args()
    main(args)

