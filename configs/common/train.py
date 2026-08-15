"""
Training-loop settings
=========================
Unlike model.py / mnist.py / optim.py, NOTHING in this file is a LazyCall
recipe -- there's no class being constructed here, just plain settings
that tools/train_net.py reads directly (e.g. `cfg.train.max_epochs`).
See docs/02_config_system_deep_dive.md if the distinction between
"LazyCall recipe" and "plain config value" isn't clear yet -- both kinds
of values can live side by side in the same config tree.

FIELD REFERENCE:
  output_dir                : where checkpoints get written
  max_epochs                : how many full passes over the training set
                                (converted to iterations internally --
                                see docs/03_training_workflow_walkthrough.md,
                                step 6)
  log_period                : print the current loss every N iterations
  num_classes                : passed to the accuracy evaluator (must match
                                 the model's own num_classes)
  checkpoint_period_epochs    : save a checkpoint every N epochs
  eval_period_epochs           : run test-set evaluation every N epochs
  mlflow.enabled                : turn MLflow logging on/off entirely
                                    (see docs/05_mlflow_and_experiment_tracking.md)
  mlflow.experiment_name          : the MLflow "experiment" this run's results
                                      group under, visible in `mlflow ui`

Every one of these can be overridden per-experiment (see
configs/MNIST/mlp_baseline.py, which overrides output_dir/max_epochs/
mlflow.experiment_name after importing this `train` object) or from the
command line, e.g. `train.max_epochs=10`.
"""
from omegaconf import OmegaConf

train = OmegaConf.create(
    dict(
        output_dir="./output/default",
        max_epochs=5,
        log_period=100,
        num_classes=10,
        checkpoint_period_epochs=1,
        eval_period_epochs=1,
        mlflow=dict(
            enabled=True,
            experiment_name="all-in-one-vision",
        ),
    )
)
