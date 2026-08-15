"""
train_net.py
=============
The single training entrypoint. This script deliberately contains almost
no logic of its own -- it just wires together pieces that live in
all_in_one_vision/: load config -> instantiate model/data/optimizer ->
build a trainer -> attach hooks -> train.

Running a different experiment (different model, different
hyperparameters) never requires editing this file -- only the
--config-file argument or its overrides change:

    python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py
    python tools/train_net.py --config-file configs/MNIST/cnn_baseline.py
    python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py \
        train.max_epochs=10 optimizer.lr=0.001
"""
import os
import sys

# Allow running as `python tools/train_net.py` from the project root
# without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from all_in_one_vision.checkpoint import Checkpointer
from all_in_one_vision.config import LazyConfig, instantiate
from all_in_one_vision.engine import (
    CheckpointHook,
    EvalHook,
    LoggingHook,
    MLflowHook,
    SimpleTrainer,
    default_argument_parser,
)
from all_in_one_vision.evaluation import AccuracyEvaluator, inference_on_dataset
from all_in_one_vision.utils import setup_logger


def main():
    args = default_argument_parser().parse_args()

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    logger = setup_logger()
    logger.info(f"Loaded config from {args.config_file}")

    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # 1) Build the model first -- the optimizer needs it (see below).
    model = instantiate(cfg.model)

    # 2) Build data loaders.
    train_loader = instantiate(cfg.dataloader.train)
    test_loader = instantiate(cfg.dataloader.test)

    # 3) Inject the real model into the optimizer recipe, THEN instantiate
    #    it. See configs/common/optim.py for why this two-step dance
    #    exists: optimizer configs must be pure/serializable data until
    #    the model actually exists.
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)

    # 4) Checkpointing (reuses fvcore.common.checkpoint.Checkpointer).
    checkpointer = Checkpointer(model, cfg.train.output_dir, optimizer=optimizer)

    # 5) Epochs -> iterations. The trainer itself only knows iterations
    #    (see engine/train_loop.py for why); this is the one place that
    #    translates the human-friendly "epochs" config into "iterations".
    iters_per_epoch = len(train_loader)
    max_iter = cfg.train.max_epochs * iters_per_epoch
    checkpoint_period = cfg.train.checkpoint_period_epochs * iters_per_epoch
    eval_period = cfg.train.eval_period_epochs * iters_per_epoch

    # 6) Evaluation function, closed over the pieces it needs.
    evaluator = AccuracyEvaluator(num_classes=cfg.train.num_classes)

    def do_eval():
        results = inference_on_dataset(model, test_loader, evaluator)
        logger.info(f"[iter {trainer.iter + 1}] Eval results: {results}")

    # 7) Build the trainer and attach hooks.
    trainer = SimpleTrainer(model, train_loader, optimizer)

    mlflow_hook = None
    if cfg.train.mlflow.enabled:
        flat_params = {
            "model.max_epochs": cfg.train.max_epochs,
            "model.target": cfg.model.get("_target_", "unknown"),
            "optimizer.lr": cfg.optimizer.lr,
            "optimizer.momentum": cfg.optimizer.get("momentum", None),
            "data.batch_size": cfg.dataloader.train.batch_size,
        }
        mlflow_hook = MLflowHook(
            flat_params=flat_params,
            experiment_name=cfg.train.mlflow.experiment_name,
            log_period=cfg.train.log_period,
        )

    trainer.register_hooks(
        [
            LoggingHook(period=cfg.train.log_period),
            EvalHook(eval_period=eval_period, eval_function=do_eval),
            CheckpointHook(checkpointer, period=checkpoint_period),
            mlflow_hook,
        ]
    )

    trainer.train(start_iter=0, max_iter=max_iter)


if __name__ == "__main__":
    main()
