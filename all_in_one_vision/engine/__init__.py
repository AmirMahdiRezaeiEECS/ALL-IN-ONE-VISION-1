"""
engine
========
The training loop (train_loop.py::SimpleTrainer) and everything that
extends it without modifying it (hooks.py). See
docs/01_architecture_and_concepts.md ("The hook pattern") for why
logging/checkpointing/evaluation/MLflow are all separate Hook classes
rather than being built into the loop itself, and
docs/03_training_workflow_walkthrough.md for a full runtime trace.
"""
from .train_loop import HookBase, TrainerBase, SimpleTrainer
from .hooks import LoggingHook, CheckpointHook, EvalHook, MLflowHook
from .defaults import default_argument_parser

__all__ = [
    "HookBase", "TrainerBase", "SimpleTrainer",
    "LoggingHook", "CheckpointHook", "EvalHook", "MLflowHook",
    "default_argument_parser",
]
