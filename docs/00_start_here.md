# Start Here

Welcome. This is the documentation for **ALL_IN_ONE_VISION_1**, a small
but "real" computer vision codebase for image classification, built to
closely follow the architecture and philosophy of
[Detectron2](https://github.com/facebookresearch/detectron2) (Meta's
object detection framework) — adapted for classification.

If you have never seen this project before, read the docs in this order:

| # | File | What you'll learn |
|---|------|--------------------|
| 1 | `00_start_here.md` (this file) | Orientation, how to run something in 2 minutes |
| 2 | `01_architecture_and_concepts.md` | The big picture: what the pieces are and why they exist |
| 3 | `02_config_system_deep_dive.md` | How `configs/*.py` files actually work under the hood |
| 4 | `03_training_workflow_walkthrough.md` | A line-by-line trace of what happens when you run `train_net.py` |
| 5 | `04_extending_the_project.md` | How to add a new dataset or model |
| 6 | `05_mlflow_and_experiment_tracking.md` | How experiment tracking is wired in |
| 7 | `06_testing_guide.md` | How the test suite is organized and what it proves |
| 8 | `07_faq_and_design_deviations.md` | Answers to "wait, why did you do it *this* way?" |

You don't have to read them all before doing anything — the fastest way
to get oriented is to run the two-minute example below, then come back
and read `01_architecture_and_concepts.md` for the "why".

---

## Who this documentation is for

Someone who:
- Knows Python and has trained a PyTorch model before (you don't need to
  be a PyTorch expert — the project already has that covered), **but**
- Has never seen this specific project, and
- May never have used Detectron2 or a "config-driven" ML codebase before.

Every doc in this folder assumes that starting point. Jargon (registry,
catalog, hook, meta-arch, LazyConfig...) is defined the first time it's
used, and re-explained briefly wherever it reappears, so you can read
these docs out of order too if you want to jump straight to one topic.

---

## The two-minute example

From the project root, with your environment active:

```bash
pip install -r requirements.txt

# Train the simplest model (an MLP) on MNIST for a few epochs.
python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py
```

That single command:
1. Reads `configs/MNIST/mlp_baseline.py` — a plain Python file describing
   *which* model, *which* dataset, and *which* hyperparameters to use.
2. Builds the real model, data loaders, and optimizer described there.
3. Trains for the configured number of epochs, printing loss every
   `train.log_period` iterations.
4. Evaluates accuracy on the MNIST test set once per epoch.
5. Saves checkpoints to `./output/mnist_mlp_v1/`.
6. Logs everything (hyperparameters, loss curve, accuracy) to MLflow.

To try the CNN instead, **you don't edit any code** — you point at a
different config:

```bash
python tools/train_net.py --config-file configs/MNIST/cnn_baseline.py
```

And to change a hyperparameter without creating a new file at all:

```bash
python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py \
    train.max_epochs=10 optimizer.lr=0.005
```

If that last command's syntax (`train.max_epochs=10`) looks unfamiliar,
that's completely expected — it's explained in depth in
`02_config_system_deep_dive.md`. For now, just notice the core idea this
whole project is built around:

> **Running a different experiment should only ever require a different
> config, never a source code change.**

Everything else in this codebase — the folder structure, the "catalog"
pattern, the hook system — exists in service of that one idea. Once you
understand *why* that idea is worth designing around, the rest of the
architecture will make a lot more sense. That's what the next doc covers.

➡️ Next: [`01_architecture_and_concepts.md`](./01_architecture_and_concepts.md)
