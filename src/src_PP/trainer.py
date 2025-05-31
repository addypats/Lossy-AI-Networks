# trainer.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainerCallback


# ==============================
# 1) Metric Function
# ==============================
def compute_classfication_metrics(eval_pred):
    """
    eval_pred: tuple (logits, labels)
    Returns a dict with 'accuracy' and 'f1'.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted")
    }


# ==============================
# 2) TTAC Callback
# ==============================
class MyClassifierCallback(TrainerCallback):
    """
    A callback that logs 'time to accuracy' thresholds and can stop
    training early once a target accuracy is reached.
    Unchanged from your original code.
    """
    def __init__(self, args=None):
        super().__init__()
        # args is a dict with keys 'report_ttac' (list of thresholds desc),
        # 'report_file' (where to write), and 'target_acc'.
        self.args = args
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)

    def on_evaluate(self, args, state, control, **kwargs):
        accuracy = kwargs["metrics"]["eval_accuracy"]
        if accuracy > self.args['target_acc']:
            print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
            control.should_training_stop = True

        for ac in self.args['report_ttac']:
            if accuracy >= ac:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {ac}, Step: {state.global_step}\n")
                break
        return super().on_evaluate(args, state, control, **kwargs)
