"""
Concrete hooks
===============
Each hook does exactly one thing and attaches to the generic loop via the
before_train/before_step/after_step/after_train extension points defined
in train_loop.HookBase. None of these hooks know about each other or
about the specifics of MNIST/MLP/CNN -- they only depend on
`self.trainer.iter`, `self.trainer.max_iter`, and
`self.trainer.latest_losses`, all of which SimpleTrainer provides
regardless of what model/dataset it's running.
"""
import logging

from .train_loop import HookBase

logger = logging.getLogger(__name__)


class LoggingHook(HookBase):
    """Print the latest loss values to the logger every `period` iterations."""

    def __init__(self, period: int = 20):
        self.period = period

    def after_step(self):
        it = self.trainer.iter
        if (it + 1) % self.period == 0 or (it + 1) == self.trainer.max_iter:
            losses = self.trainer.latest_losses
            losses_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
            logger.info(f"iter {it + 1}/{self.trainer.max_iter}  {losses_str}")


class CheckpointHook(HookBase):
    """Save a checkpoint every `period` iterations (and at the end of training)."""

    def __init__(self, checkpointer, period: int):
        self.checkpointer = checkpointer
        self.period = period

    def after_step(self):
        it = self.trainer.iter
        if (it + 1) % self.period == 0 or (it + 1) == self.trainer.max_iter:
            self.checkpointer.save(f"model_iter_{it + 1:07d}")

    def after_train(self):
        self.checkpointer.save("model_final")


class EvalHook(HookBase):
    """Run `eval_function()` every `eval_period` iterations (and at the end)."""

    def __init__(self, eval_period: int, eval_function):
        self.eval_period = eval_period
        self.eval_function = eval_function

    def after_step(self):
        it = self.trainer.iter
        if self.eval_period <= 0:
            return
        if (it + 1) % self.eval_period == 0 or (it + 1) == self.trainer.max_iter:
            self.eval_function()


class MLflowHook(HookBase):
    """
    Logs config params at the start of training, per-step losses
    periodically, and the final checkpoint as an artifact at the end.

    Kept as a hook (not baked into the trainer) specifically so training
    works with or without MLflow available/enabled -- toggled from config
    via `train.mlflow.enabled`.
    """

    def __init__(self, flat_params: dict, experiment_name: str, log_period: int = 20,
                 artifact_path: str = None):
        self.flat_params = flat_params
        self.experiment_name = experiment_name
        self.log_period = log_period
        self.artifact_path = artifact_path

    def before_train(self):
        import mlflow

        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()
        mlflow.log_params(self.flat_params)

    def after_step(self):
        it = self.trainer.iter
        if (it + 1) % self.log_period == 0:
            import mlflow

            mlflow.log_metrics(self.trainer.latest_losses, step=it + 1)

    def after_train(self):
        import mlflow

        if self.artifact_path:
            mlflow.log_artifact(self.artifact_path)
        mlflow.end_run()
