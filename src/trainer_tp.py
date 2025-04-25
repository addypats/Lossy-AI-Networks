from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

class MyClassifierCallback(TrainerCallback):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.args['report_ttac'] = sorted(self.args['report_ttac'], reverse=True)

    def on_evaluate(self, args, state, control, **kwargs):
        accuracy = kwargs["metrics"].get("eval_accuracy", 0)
        if accuracy > self.args['target_acc']:
            print(f"Target accuracy {self.args['target_acc']} reached. Stopping training.")
            control.should_training_stop = True

        for ac in self.args['report_ttac']:
            if accuracy >= ac:
                with open(self.args['report_file'], "a") as f:
                    f.write(f"Accuracy: {accuracy:.3f}, Threshold: {ac},  Step: {state.global_step}\n")
                break
        return super().on_evaluate(args, state, control, **kwargs)

def compute_classfication_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }