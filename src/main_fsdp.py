
# Original Pegah's code
# from comms import LossyNetwork, GillbertElliotLossyNetwork
from comms_fsdp_dist import LossyNetwork, GillbertElliotLossyNetwork
from trainer_fsdp import MyClassifierCallback, compute_exact_match_metric, compute_classfication_metrics, FSDPProbeCallback
from data import get_dataset
from transformers import TrainingArguments, Trainer, TrainerCallback
import os
import pandas as pd
import yaml
from models import get_classifier_and_tokenizer
import fsdp_introspect
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d  # useful to wrap internal entrypoints too
import time
# from lossy_patch_packets_1 import install_lossy_collectives
from lossy_patch_sanity_check import install_lossy_collectives
import inspect

# print("install_lossy_collectives sig:", inspect.signature(lossy_patch.install_lossy_collectives))

classification_datasets = ['winogrande', 'mnli', 'sst2', 'hellaswag', 'piqa', 'arc', 'quality']
generation_datasets = ['hotpotqa', 'squad', 'tinysquad', 'newsqa', 'triviaqa']


class LossyStepBump(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        # HF keeps state.global_step consistent across ranks.
        os.environ["LOSSY_GLOBAL_STEP"] = str(state.global_step)


class LossyGradHookCallback(TrainerCallback):
    """
    Installs lossy gradient hooks on the (possibly FSDP-wrapped) model
    at the beginning of training.
    """
    def __init__(self, lossy: LossyNetwork, enable: bool = False, include_bias: bool = False):
        super().__init__()
        self.lossy = lossy
        self.enable = enable
        self.include_bias = include_bias
        self._installed = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enable or self._installed:
            return
        model = kwargs.get("model", None)
        if model is None:
            return
        attach_lossy_grad_hooks(model, self.lossy, include_bias=self.include_bias, verbose=True)
        self._installed = True




# # --- BEGIN INTROSPECTION WRAPPER ---

# def _rk():
#     try:
#         return dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
#     except Exception:
#         return -1

# class _FileLogger:
#     def __init__(self, path):
#         self.path = path
#         os.makedirs(os.path.dirname(self.path), exist_ok=True)
#     def log(self, msg):
#         # print(msg, flush=True)
#         with open(self.path, "a") as f:
#             f.write(msg + "\n")

# def _wrap(obj, fn_name, logger):
#     if not hasattr(obj, fn_name):
#         return
#     orig = getattr(obj, fn_name)
#     if not callable(orig):
#         return
#     # prevent double-wrapping
#     if getattr(orig, "__wrapped_by_introspect__", False):
#         return
#     def wrapped(*args, **kwargs):
#         t0 = time.time()
#         try:
#             out = orig(*args, **kwargs)
#             return out
#         finally:
#             dt = (time.time() - t0) * 1000.0
#             logger.log(f"[INTROSPECT] rank={_rk()} {getattr(obj,'__name__',obj.__class__.__name__)}.{fn_name} dt={dt:.2f} ms")
#     wrapped.__wrapped_by_introspect__ = True
#     setattr(obj, fn_name, wrapped)

# def install_collective_hooks(logdir="logs"):
#     if not (dist.is_available() and dist.is_initialized()):
#         # print("[INTROSPECT] dist not initialized; skipping install", flush=True)
#         return
#     rank = _rk()
#     logger = _FileLogger(os.path.join(logdir, f"rank{rank}_collectives.txt"))
#     try:
#         # public APIs Accelerate/HF use
#         for fn in ["all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor", "barrier"]:
#             _wrap(dist, fn, logger)
#         # low-level entrypoints (used under the hood)
#         for fn in ["_allgather_base", "_reduce_scatter_base", "all_gather_into_tensor", "reduce_scatter_tensor"]:
#             _wrap(c10d, fn, logger)
#         logger.log(f"[INTROSPECT] rank={rank} hooks installed")
#     except Exception as e:
#         logger.log(f"[INTROSPECT] rank={rank} FAILED to install hooks: {e}\n{traceback.format_exc()}")

# # --- END INTROSPECTION WRAPPER ---


def main(args):

    with open("src/dataset_config.yaml") as config:
        try:
            dataset_config = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = dataset_config[args.dataset]
    loss_type = args.loss_type
    if loss_type == 'ber':
        network = LossyNetwork(args)
        network.set_seed(args.seed)
    elif loss_type == 'g-e':
        configs = pd.read_csv('g_e_params.csv')
        # configs = pd.read_csv('ge_params_burst.csv')
        # configs = pd.read_csv('ge_params_questions.csv')
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
        network.set_seed(args.seed)
    
    elif loss_type == 'det':
        import pandas as pd
        from deterministic_loss import DeterministicBurstLossyNetwork
        # Deterministic, GE-free, CSV-driven burst loss model
        # We keep using args.ge_config as the run_id to avoid changing your launch scripts.
        configs = pd.read_csv('runs_profile_det.csv')
        run_id    = args.det_config        # e.g., "A3", "B2"
        model_name = args.model_name
        task       = args.dataset
        seed       = args.seed
        nodes      = args.num_nodes

        # Load the CSV with all deterministic experiment configs
        # df = pd.read_csv(csv_path)

        # Helper casting functions
        def to_int(x):   return int(x)
        def to_float(x): return float(x)
        def to_str(x):   return str(x)

         # Find matching row by run_id (the column name in your CSV)
        row = configs.loc[configs["runs_id"] == run_id]
        if row.empty:
            raise ValueError(f"id '{runs_id}' not found in {configs}. "
                             f"Available: {configs['runs_id'].tolist()}")
        r = row.iloc[0]   # <-- define r before using it

        # import os

        run_id_str = str(r["runs_id"])        # e.g., "high_persistence_low_intensity_1"

        # Create a subfolder inside det_logs/
        log_dir = os.path.join("det_logs", run_id_str)
        os.makedirs(log_dir, exist_ok=True)

        # Instantiate directly from CSV; no pandas needed and no recomputation inside the class.
        # det_config = configs[configs['id'] == args.det_config].iloc[0]
        network = DeterministicBurstLossyNetwork.from_params(
            run_id    = to_str(r["runs_id"]),
            T_steps   = to_int(r["T_steps"]),
            N         = to_int(r["N"]),
            L_overall = to_float(r["L_overall"]),
            lrg       = to_float(r["lrg"]),
            lrb       = to_float(r["lrb"]),
            piB       = to_float(r["piB"]),
            B         = to_int(r["B"]),
            Eb        = to_float(r["Eb"]),
            rho       = to_float(r["rho"]),
            seed      = to_int(r["seed"]),
            skew_frac = to_float(r["skew_frac"]),
            gap_mode  = to_str(r["gap_mode"]),
            log_dir   = log_dir,
            strict_validate = True
        )

    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    network.set_seed(args.seed)

    # lossy=LossyNetwork(args)

    # lossy=network
    # os.environ["LOSS_TYPE"] = loss_type
    # os.environ["LOSS_RATE"] = str(getattr(lossy, "loss_rate", args.loss_rate))      # e.g., "0.01" = 1%
    # os.environ["LOSSY_GLOBAL_STEP"] = "0"                   # updated by callback each step
    # os.environ["LOSSY_CALL_COUNTER"] = "0"                  # per-process, used for unique masks

    # # optional: make randomness per-rank reproducible
    # try:
    #     dist_inited = dist.is_available() and dist.is_initialized()
    #     rank = dist.get_rank() if dist_inited else int(os.environ.get("RANK", "0"))
    # except Exception:
    #     rank = 0
    # base_seed = getattr(args, "seed", 0)
    # lossy.set_seed(base_seed + rank)

    # enable_ag = args.loss_enable_all or args.loss_enable_ag
    # enable_rs = args.loss_enable_all or args.loss_enable_rs
    # enable_ar = args.loss_enable_all or args.loss_enable_ar

    # # 1) install BEFORE building model/accelerator/trainer
    # install_lossy_collectives(lossy,
    #                           enable_allgather=enable_ag,
    #                           enable_rs=enable_rs,
    #                           enable_allreduce=enable_ar,
    #                           min_numel=0)
    
    
    # for rank aware strategy with input gpu getting loss for dist training on multiple instances
    
    # Use `network` everywhere as the lossy object
    lossy = network

    # Optional: env vars for logging / consistency with older code
    os.environ["LOSS_TYPE"] = loss_type
    os.environ["LOSS_RATE"] = str(getattr(lossy, "loss_rate", args.loss_rate))  # e.g., "0.01"
    os.environ["LOSSY_GLOBAL_STEP"] = "0"          # updated by LossyStepBump
    os.environ["LOSSY_CALL_COUNTER"] = "0"         # per-process, if you ever need it

    # ---- Per-rank loss RNG seed (for inter-run variability on loss) ----
    try:
        dist_inited = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_inited else int(os.environ.get("RANK", "0"))
    except Exception:
        rank = 0

    # base_seed = getattr(args, "seed", 0)
    # # loss seed = base_seed + rank  (you can later separate training_seed vs loss_seed)
    # lossy.set_seed(base_seed + rank)
    
    if loss_type == 'det':
    # Do nothing! Let the seed from the CSV handle it.
        pass 
    else:
        # For BER or G-E, use a global seed (same for all ranks)
        lossy.set_seed(args.seed)
    
    # -------------------------------------------------------------------

    enable_ag = args.loss_enable_all or args.loss_enable_ag
    enable_rs = args.loss_enable_all or args.loss_enable_rs
    enable_ar = args.loss_enable_all or args.loss_enable_ar

    # 2A: install rank-aware lossy collectives
    # IMPORTANT: this must be called BEFORE building model/Trainer/FSDP
    install_lossy_collectives(
        loss=lossy,
        enable_allgather=enable_ag,
        enable_rs=enable_rs,
        enable_allreduce=enable_ar,
        min_numel=0,
        num_nodes=args.num_nodes,   # <-- NEW: tell the wrapper how many nodes there are
        gpus_per_node=args.gpus_per_node, # <-- NEW: tell the wrapper how many GPUs per node (for rank-aware masking)
    )


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
        os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f) # for reproducibility

    args.output_dir = output_dir

    callback_args = { # report time to accuracy #TODO change this to "steps to accuracy"
        'report_ttac' : dataset_config['report_ttac'],
        'report_file' : f"{args.output_dir}/ttac_report.txt",
        'target_acc': dataset_config['target_acc'],
    }
    #if args.dataset in generation_datasets:
    #    callback_args['eos_token_id'] = tokenizer.eos_token_id
    #    compute_metrics = compute_exact_match_metric(tokenizer)
    #    callback = MyQACallback(callback_args)
    #    trainer_class = MyQATrainer
    #    callback = MyQACallback(callback_args)
    #    trainer_class = MyQATrainer
    if args.dataset in generation_datasets:
        # callback_args['eos_token_id'] = tokenizer.eos_token_id
        compute_metrics = compute_exact_match_metric(tokenizer)
        # reuse classifier-style callback; it will watch EM instead of accuracy
        callback = MyClassifierCallback(callback_args)
        trainer_class = Trainer
    else:
        compute_metrics = compute_classfication_metrics
        callback = MyClassifierCallback(callback_args)
        trainer_class = Trainer
   
    is_qa = args.dataset in generation_datasets

    eval_bs = args.batch_size // 2 if is_qa else args.batch_size

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_bs,
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
        # -------- FSDP: the important bits --------
        fsdp="full_shard",              # shard params+grads+optstates
        fsdp_config={
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "use_orig_params": True,
            "forward_prefetch": True,
            },
                # optional but useful:                        # avoid wrapping tiny modules
        remove_unused_columns=False,
        gradient_checkpointing=False,                          # extra memory headroom
        ddp_find_unused_parameters=False,                     # usually better with FSDP
        ddp_backend="nccl",
        
        # 🔹 KEY NEW BITS FOR QA
        eval_accumulation_steps=1,      # flush eval tensors to CPU every step
    )

    # optimizer = AdamW
    # optimizer = None

    # fsdp_introspect.enable(model, optimizer)
    # fsdp_introspect.enable(model, optimizer, log_dir="logs/fsdp_probe", flush_every=64)

    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[callback, LossyStepBump()],
        compute_metrics=compute_metrics,
    )

    # If incase you want to see a probe in FSDP here is what you replace callbacks in trainer with:
    # callbacks=[callback, FSDPProbeCallback(), LossyStepBump()],

    # fsdp_introspect.attach_trainer(trainer)

    # install_collective_hooks(logdir="logs_reduce")

    trainer.train()

    # fsdp_introspect.finalize()   # writes *_sorted.jsonl and *_sorted.csv

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed Training with Packet Loss")
    parser.add_argument('--num_nodes', type=int, default=2, help='Number of nodes')
    parser.add_argument('--loss_rate', type=float, default=0.001, help='Packet loss rate When using Bernoulli')
    parser.add_argument('--loss_type', type=str, default='ber', choices=['ber', 'g-e', 'det'], help='Type of packet loss simulation: "ber" for Bernoulli, "g-e" for Gilbert-Elliott')
    parser.add_argument('--ge_config', type = str, default = 'default', help='configuration id for Gilbert-Elliott loss simulation. Refer to g_e_params.csv')
    parser.add_argument("--det_config", type=str, default="default", help="configuration id for Deterministic Custom loss simulation. Refer to runs_profiles_det.csv)")
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
    
    # number of GPUs per node for dist training setup
    parser.add_argument('--gpus_per_node', type=int, default=4, help='Physical GPUs per server')
    
    # ... your existing argparse setup ...
    parser.add_argument("--loss-enable-ag", action="store_true",
                    help="Inject loss into all_gather_into_tensor.")
    parser.add_argument("--loss-enable-rs", action="store_true",
                    help="Inject loss into reduce_scatter_tensor.")
    parser.add_argument("--loss-enable-ar", action="store_true",
                    help="Inject loss into all_reduce (misc syncs).")
    # optional: one switch to turn them all on
    parser.add_argument("--loss-enable-all", action="store_true",
                    help="Shortcut: enable AG, RS, and AR together.")

    args = parser.parse_args()
    
    main(args)

