"""
evaluation
============
How model quality gets measured. DatasetEvaluator (evaluator.py) is an
interface -- process() a batch, evaluate() once at the end -- so a new
metric can be added as a new evaluator class without touching the
training loop. AccuracyEvaluator is the one implementation used so far.
"""
from .evaluator import DatasetEvaluator, inference_on_dataset
from .classification_evaluation import AccuracyEvaluator

__all__ = ["DatasetEvaluator", "inference_on_dataset", "AccuracyEvaluator"]
