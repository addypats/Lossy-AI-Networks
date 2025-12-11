# trainer.py

from transformers import Trainer, TrainerCallback
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ---- Optional: prints what FSDP wrapped and shard sizes per rank ----
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _HAS_FSDP = True
except Exception:
    _HAS_FSDP = False

class FSDPProbeCallback(TrainerCallback):
    """
    On train begin, list FSDP-wrapped modules (e.g., one per LlamaDecoderLayer)
    and print local-shard vs full parameter sizes for each wrapped unit.
    Safe to keep enabled during real finetuning; it only prints once.
    """
    def on_train_begin(self, args, state, control, **kwargs):
        if not _HAS_FSDP:
            return
        model = kwargs["model"]
        import os
        rank = int(os.environ.get("RANK", "0"))

        wrapped = []
        for name, mod in model.named_modules():
            if isinstance(mod, FSDP):
                wrapped.append(name if name else "<root>")

        # print(f"[R{rank}] FSDP-wrapped modules ({len(wrapped)}):")
        for w in wrapped:
            #print(f"[R{rank}]  - {w}")
            continue

        # Print shard vs full sizes per wrapped unit
        for name, mod in model.named_modules():
            if isinstance(mod, FSDP):
                local_elems = sum(p.numel() for p in mod.parameters(recurse=False))
                from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
                with _FSDP.summon_full_params(mod, writeback=False, recurse=False):
                    full_elems = sum(p.numel() for p in mod.parameters(recurse=False))
                #print(f"[R{rank}] {name or '<root>'}: local={local_elems} elems, full={full_elems} elems")
                continue

# # ---- Your metrics (unchanged) ----
# def compute_classfication_metrics(eval_pred):
#     logits, labels = eval_pred
#     preds = np.argmax(logits, axis=1)
#     # debug peek
#     if len(labels) > 0:
#         try:
#             print("compute_classfication_metrics: true vs pred (first 5):",
#                   list(zip(labels[:5], preds[:5])))
#         except Exception:
#             pass
#     return {
#         "accuracy": accuracy_score(labels, preds),
#         "f1": f1_score(labels, preds, average="weighted"),
#     }



# compute metrics for squad

import evaluate
import numpy as np

def compute_exact_match_metric(tokenizer):
    squad_metric = evaluate.load("squad")  # or 'squad_v2'

    def _compute(eval_pred):
        logits, labels = eval_pred  # logits: (batch, seq, vocab), labels: (batch, seq)

        # Argmax over vocab dimension to get predicted token IDs
        # Shape: (batch, seq_len)
        # pred_ids = np.argmax(logits, axis=-1)

        # Convert to numpy in case HF gives torch tensors
        if not isinstance(logits, np.ndarray):
            logits_np = np.array(logits)
        else:
            logits_np = logits

        if not isinstance(labels, np.ndarray):
            labels_np = np.array(labels)
        else:
            labels_np = labels

        # Argmax over vocab dimension to get predicted token IDs: (batch, seq_len)
        if logits_np.ndim == 3:
            pred_ids = np.argmax(logits_np, axis=-1)
        else:
            # Just in case predictions are already token IDs
            pred_ids = logits_np

        squad_preds = []
        squad_refs = []

        batch_size = labels.shape[0]
        for i in range(batch_size):
            label_ids = labels[i]
            pred_ids_i = pred_ids[i]

            # Positions where we have real labels (answer tokens), not -100
            mask = label_ids != -100
            if not np.any(mask):
                # No answer tokens for this example (shouldn't really happen
                # if preprocessing is correct), skip it.
                continue

            true_answer_ids = label_ids[mask]
            pred_answer_ids = pred_ids_i[mask]

            true_text = tokenizer.decode(true_answer_ids, skip_special_tokens=True).strip()
            pred_text = tokenizer.decode(pred_answer_ids, skip_special_tokens=True).strip()

            squad_preds.append({
                "id": str(i),
                "prediction_text": pred_text,
            })
            squad_refs.append({
                "id": str(i),
                "answers": {"text": [true_text], "answer_start": [0]},
            })

        # Compute EM/F1
        results = squad_metric.compute(
            predictions=squad_preds,
            references=squad_refs
        )
        # results has 'exact_match' and 'f1'
        return results

    return _compute


# ---- Your early-stop callback (kept) ----
class MyClassifierCallback(TrainerCallback):
    def __init__(self, args=None):
        super().__init__()
        self.counter = 0
        self.patience = 2
        self.args = args or {}
        # normalize args
        self.args.setdefault('report_ttac', [])
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)
        self.args.setdefault('report_file', 'ttac_report.txt')
        self.args.setdefault('target_acc', 0.0)
        # ensure report file exists
        try:
            with open(self.args['report_file'], "r"):
                pass
        except FileNotFoundError:
            with open(self.args['report_file'], "w") as f:
                f.write("")

    def on_evaluate(self, args, state, control, **kwargs):
        # accuracy = kwargs["metrics"].get("eval_accuracy")
        # if accuracy is None:
        #     return super().on_evaluate(args, state, control, **kwargs)
        
        # to make it suitable for squad too
        metrics = kwargs.get("metrics", {})
        # Prefer eval_accuracy; fall back to eval_exact_match
        accuracy = metrics.get("eval_accuracy", None)
        if accuracy is None:
            accuracy = metrics.get("eval_exact_match", None)

        if accuracy is None:
            return super().on_evaluate(args, state, control, **kwargs)

        if accuracy > self.args['target_acc']:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
                with open(self.args['report_file'], "w") as f:
                    f.write(
                        f"Accuracy: {accuracy:.3f}, "
                        f"Threshold: {self.args['report_ttac'][0] if self.args['report_ttac'] else 'N/A'}, "
                        f"Step: {state.global_step}\n"
                    )
                control.should_training_stop = True

        return super().on_evaluate(args, state, control, **kwargs)


        if accuracy > self.args['target_acc']:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
                with open(self.args['report_file'], "w") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {self.args['report_ttac'][0] if self.args['report_ttac'] else 'N/A'}, Step: {state.global_step}\n")
                control.should_training_stop = True

        return super().on_evaluate(args, state, control, **kwargs)

