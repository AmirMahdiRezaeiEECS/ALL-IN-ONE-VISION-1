"""
AccuracyEvaluator
===================
REUSE-FIRST: torchmetrics.Accuracy already implements correct, tested
multiclass accuracy (including edge cases like class imbalance handling
via `average=`). We just accumulate batches into it and read out the
final number -- no custom accuracy math.
"""
from torchmetrics import Accuracy

from .evaluator import DatasetEvaluator


class AccuracyEvaluator(DatasetEvaluator):
    def __init__(self, num_classes: int):
        self.metric = Accuracy(task="multiclass", num_classes=num_classes)

    def reset(self):
        self.metric.reset()

    def process(self, targets, outputs):
        preds = outputs.argmax(dim=1)
        self.metric.update(preds, targets)

    def evaluate(self):
        return {"accuracy": self.metric.compute().item()}
