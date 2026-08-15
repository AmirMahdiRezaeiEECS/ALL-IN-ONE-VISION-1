# Training Workflow Walkthrough

This document traces `tools/train_net.py` top to bottom, connecting each
line to the concepts explained in the previous two documents. If you
understand every line of `train_net.py` by the end of this doc, you
understand how the whole project fits together at runtime.

Open `tools/train_net.py` in an editor alongside this doc — the section
headers below match its inline numbered comments (`# 1) ...`, `# 2) ...`).

## Table of contents
- [Before main() even runs](#before-main-even-runs)
- [1) Parse arguments, load config](#1-parse-arguments-load-config)
- [2) Build the model](#2-build-the-model)
- [3) Build the data loaders](#3-build-the-data-loaders)
- [4) Build the optimizer](#4-build-the-optimizer)
- [5) Set up checkpointing](#5-set-up-checkpointing)
- [6) Convert epochs to iterations](#6-convert-epochs-to-iterations)
- [7) Define the evaluation function](#7-define-the-evaluation-function)
- [8) Build the trainer and attach hooks](#8-build-the-trainer-and-attach-hooks)
- [9) Train](#9-train)
- [What actually happens inside trainer.train()](#what-actually-happens-inside-trainertrain)
- [A complete worked run, iteration by iteration](#a-complete-worked-run)

---

## Before `main()` even runs

At the top of `train_net.py`:
```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```
This just makes sure the project root is importable as `all_in_one_vision`
and `configs`, whether you run the script as `python tools/train_net.py`
from the project root or from somewhere else. If you install the project
properly (`pip install -e .`) later, this line becomes unnecessary — it's
a convenience for running straight from a git checkout.

---

## 1) Parse arguments, load config

```python
args = default_argument_parser().parse_args()
cfg = LazyConfig.load(args.config_file)
cfg = LazyConfig.apply_overrides(cfg, args.opts)
```

`default_argument_parser()` (in `engine/defaults.py`) is a tiny, reusable
`argparse` setup accepting `--config-file` and a list of `key=value`
override strings. See
[`02_config_system_deep_dive.md`](./02_config_system_deep_dive.md) for
exactly what `load()` and `apply_overrides()` do. After this, `cfg` is a
fully-resolved (overrides applied), but still entirely **inert** —
nothing has been built yet — config tree.

## 2) Build the model

```python
model = instantiate(cfg.model)
```

This is the first point where something real gets constructed: `cfg.model`
(a `LazyCall` recipe naming e.g. `SimpleMLPClassifier`) becomes an actual
`nn.Module`, with randomly initialized weights. It's built *before* the
data loaders and optimizer specifically because the optimizer needs it —
see step 4.

## 3) Build the data loaders

```python
train_loader = instantiate(cfg.dataloader.train)
test_loader = instantiate(cfg.dataloader.test)
```

Each of these resolves down through `build_loader()` (in `data/build.py`),
which looks up `"mnist_train"` / `"mnist_test"` in the `DatasetCatalog`,
builds the transform (also a nested recipe), and returns a real
`torch.utils.data.DataLoader`. First time this runs against real MNIST
(not the fake dataset the tests use), torchvision downloads MNIST to
`./datasets/mnist/` automatically.

## 4) Build the optimizer

```python
cfg.optimizer.params.model = model
optimizer = instantiate(cfg.optimizer)
```

This is the "inject a real object into the config" pattern explained in
detail in the config deep-dive doc's
[optimizer special case](./02_config_system_deep_dive.md#the-optimizer-special-case)
section. Short version: the optimizer's `params` argument needs
`model.parameters()`, and the model didn't exist when the config file
was written — so we patch the real model into the config right before
building the optimizer.

## 5) Set up checkpointing

```python
checkpointer = Checkpointer(model, cfg.train.output_dir, optimizer=optimizer)
```

`Checkpointer` (in `checkpoint/checkpointer.py`) is a thin subclass of
`fvcore`'s `Checkpointer`. Passing both `model` and `optimizer` means
saved checkpoints include optimizer state too (momentum buffers, etc.),
so training could in principle be resumed exactly, not just restarted
with random optimizer state.

## 6) Convert epochs to iterations

```python
iters_per_epoch = len(train_loader)
max_iter = cfg.train.max_epochs * iters_per_epoch
checkpoint_period = cfg.train.checkpoint_period_epochs * iters_per_epoch
eval_period = cfg.train.eval_period_epochs * iters_per_epoch
```

The training loop itself (`SimpleTrainer`) only knows about
**iterations** (one iteration = one batch), not epochs — matching
Detectron2's convention. This is the one place that translates the
human-friendly `train.max_epochs` from your config into the iteration
count the loop actually uses. `len(train_loader)` is "how many batches
make up one full pass over the training set", so multiplying by that
converts epochs → iterations for anything that should happen "once per
epoch" (checkpointing, evaluation).

## 7) Define the evaluation function

```python
evaluator = AccuracyEvaluator(num_classes=cfg.train.num_classes)

def do_eval():
    results = inference_on_dataset(model, test_loader, evaluator)
    logger.info(f"[iter {trainer.iter + 1}] Eval results: {results}")
```

`do_eval` is a plain closure capturing `model`, `test_loader`, and
`evaluator` from the surrounding scope. It's defined here (rather than as
a standalone function) specifically so it can be handed to `EvalHook`
below without needing to pass all three objects through the hook's
constructor separately.

## 8) Build the trainer and attach hooks

```python
trainer = SimpleTrainer(model, train_loader, optimizer)

trainer.register_hooks([
    LoggingHook(period=cfg.train.log_period),
    EvalHook(eval_period=eval_period, eval_function=do_eval),
    CheckpointHook(checkpointer, period=checkpoint_period),
    mlflow_hook,   # None if MLflow is disabled in this config
])
```

`SimpleTrainer` itself only knows how to do one thing: pull a batch,
forward, backward, step (see `engine/train_loop.py`). Everything else —
when to print progress, when to save, when to evaluate, whether to talk
to MLflow — is a `Hook`, attached here, in one readable list. Notice
`register_hooks` silently skips `None` entries, which is why
`mlflow_hook` can simply be `None` when `cfg.train.mlflow.enabled` is
`False`, with no `if` needed at the call site.

## 9) Train

```python
trainer.train(start_iter=0, max_iter=max_iter)
```

This call blocks until training finishes. Everything below is what
happens *inside* it.

---

## What actually happens inside `trainer.train()`

From `engine/train_loop.py::TrainerBase.train`:

```python
self._call_hooks("before_train")            # MLflowHook starts an MLflow run here
for self.iter in range(start_iter, max_iter):
    self._call_hooks("before_step")           # (no hooks use this one currently)
    self.run_step()                            # SimpleTrainer.run_step(): the actual math
    self._call_hooks("after_step")             # Logging/Eval/Checkpoint/MLflow hooks check
                                                 #   "is it my turn?" and act if so
self._call_hooks("after_train")              # CheckpointHook saves "model_final";
                                                #   MLflowHook ends the run
```

And `SimpleTrainer.run_step()` (the actual training math, unchanged by
which hooks are attached):

```python
self.model.train()
images, targets = self._next_batch()             # re-starts the loader at epoch boundaries

loss_dict = self.model(images, targets)            # the forward contract: dict of named losses
losses = sum(loss_dict.values())

self.optimizer.zero_grad()
losses.backward()
self.optimizer.step()

self.latest_losses = {...}                          # hooks read this to log/print
```

---

## A complete worked run

Concretely, for `configs/MNIST/mlp_baseline.py`'s defaults (`max_epochs=5`,
MNIST has 60,000 training images, `batch_size=64`):

- `iters_per_epoch = 60000 / 64 ≈ 937`
- `max_iter = 5 * 937 = 4685`
- `checkpoint_period = eval_period = 1 * 937 = 937` (once per epoch, since
  `checkpoint_period_epochs` / `eval_period_epochs` both default to `1`)

So training runs iterations `0` through `4684`. Every 100 iterations
(`train.log_period`), `LoggingHook` prints the current loss. Every 937
iterations, `CheckpointHook` saves a checkpoint and `EvalHook` runs a
full pass over the test set and logs accuracy. At iteration 4684 (the
last one), both of those also fire (they check `it+1 == max_iter` as
well as the periodic condition), plus `CheckpointHook.after_train` saves
one final `model_final.pth`, and `MLflowHook.after_train` closes out the
MLflow run.

Want to see all of this run against a tiny synthetic dataset (no MNIST
download, finishes in under a second)? That's exactly what
`tests/test_config_pipeline.py::test_full_pipeline_smoke` does — see
[`06_testing_guide.md`](./06_testing_guide.md).

➡️ Next: [`04_extending_the_project.md`](./04_extending_the_project.md)
