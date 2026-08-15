# MLflow & Experiment Tracking

## Why experiment tracking exists as a requirement at all

Once you can run many experiments cheaply (the whole point of the config
system — see `01_architecture_and_concepts.md`), you very quickly lose
track of *which config produced which result* if you're not recording it
somewhere. MLflow is an established, open-source tool for exactly this:
it records, per training run, the hyperparameters used, the metrics over
time (loss, accuracy), and any output artifacts (like a checkpoint file),
all browsable later in a web UI. We integrate it rather than building our
own tracking (reuse-first).

## Where it lives in the codebase

Everything MLflow-specific is contained in one class:
`all_in_one_vision/engine/hooks.py::MLflowHook`. This is deliberate —
see the [hook pattern](./01_architecture_and_concepts.md#4-the-hook-pattern-extending-a-loop-without-editing-it)
in the architecture doc. Because it's "just a hook", MLflow can be:
- turned off entirely (`train.mlflow.enabled = False` in config) with
  zero effect on training itself,
- understood in isolation, without reading any other part of `engine/`.

## What gets logged, and when

| MLflow call | When | What |
|---|---|---|
| `mlflow.start_run()` | `before_train` (once, at the very start) | Begins tracking this run |
| `mlflow.log_params(...)` | `before_train` | The hyperparameters `train_net.py` collected: model class, max_epochs, learning rate, momentum, batch size |
| `mlflow.log_metrics(...)` | `after_step`, every `train.log_period` iterations | The current loss values from `trainer.latest_losses` |
| `mlflow.log_artifact(...)` | `after_train` (once, at the very end) | The output directory / final checkpoint, if `artifact_path` was set |
| `mlflow.end_run()` | `after_train` | Closes out the run |

## Running it yourself

```bash
python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py
mlflow ui   # opens a local web UI at http://127.0.0.1:5000
```

Each run shows up under the experiment name set in your config
(`train.mlflow.experiment_name`, e.g. `"all-in-one-vision-mnist"`). You
can compare loss curves and final accuracy across every run you've
launched with different configs/overrides, without having written a
single line of comparison code yourself.

## A version-specific gotcha, not our bug

Recent MLflow releases (3.x) put the bare local filesystem tracking store
(the default, `"./mlruns"`) into "maintenance mode" and raise an error
unless you either:
- set the environment variable `MLFLOW_ALLOW_FILE_STORE=true`, or
- point at a database-backed store instead, e.g.
  `mlflow.set_tracking_uri("sqlite:///mlflow.db")` (would need to be
  added inside `MLflowHook.before_train` if you want this as the
  project's default — currently left as a manual step so you can choose
  based on your own MLflow version).

This surfaces the first time you actually run training with MLflow
enabled and your installed `mlflow` is new enough to have this
restriction — it's a property of MLflow itself, not a bug in this
project's integration.

## Extending it

Want to log something new (e.g. a confusion matrix image, or per-class
accuracy)? Add it to `MLflowHook` — most likely in `after_step` (for
per-iteration data) or `after_train` (for a final summary). Since
`MLflowHook` already has access to `self.trainer` (set automatically when
a hook is registered — see `engine/train_loop.py::TrainerBase.register_hooks`),
you can pull anything the trainer tracks (`self.trainer.iter`,
`self.trainer.latest_losses`, `self.trainer.model`, ...) without changing
its constructor.

➡️ Next: [`06_testing_guide.md`](./06_testing_guide.md)
