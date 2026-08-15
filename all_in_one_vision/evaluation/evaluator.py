"""
DatasetEvaluator
==================
Same interface as Detectron2's evaluation/evaluator.py: an evaluator
accumulates per-batch predictions via `process()`, and `evaluate()`
produces the final metric dict once all batches have been seen.

Keeping this as an interface (rather than hardcoding "compute accuracy"
inside the eval loop) is what lets a future evaluator -- e.g. a
per-class-accuracy or a confusion-matrix evaluator -- be swapped in via
config without touching inference_on_dataset() or the training loop.
"""
import torch


class DatasetEvaluator:
    def reset(self):
        """Clear any accumulated state. Called once before evaluation starts."""
        pass

    def process(self, targets, outputs):
        """Called once per batch with the ground-truth targets and model outputs."""
        pass

    def evaluate(self):
        """Called once after all batches are processed. Returns a metrics dict."""
        pass


def inference_on_dataset(model, data_loader, evaluator: DatasetEvaluator):
    """
    Run `model` over every batch in `data_loader` in eval mode, feed each
    batch's (targets, outputs) to `evaluator`, and return the final
    metrics dict.
    """
    model.eval()
    evaluator.reset()
    with torch.no_grad():
        for images, targets in data_loader:
            outputs = model(images)
            evaluator.process(targets, outputs)
    return evaluator.evaluate()
