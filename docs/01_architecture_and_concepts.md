# Architecture & Core Concepts

This is the most important document in this folder. It explains the
*shape* of the project — what each top-level piece is responsible for,
why it's separate from the others, and how they connect. Once this
clicks, every individual file in the codebase should feel like it's
"obviously" in the right place.

## Table of contents
- [Why this project looks the way it does](#why-this-project-looks-the-way-it-does)
- [The big picture (data flow diagram)](#the-big-picture)
- [Package-by-package tour](#package-by-package-tour)
- [Five core concepts, explained from scratch](#five-core-concepts-explained-from-scratch)
- [Why classification-only, and what was deliberately left out](#why-classification-only)

---

## Why this project looks the way it does

Imagine the simplest possible way to write a classifier training script:
one `train.py` file that hardcodes the dataset, the model, the optimizer,
and the training loop, all in one place. That's genuinely the *right*
choice for a one-off experiment (it's exactly what this project's own
"v1" MLP script looked like, before this restructuring).

It stops being the right choice the moment you want to:
- try a second model (CNN) without duplicating the whole script,
- try a second dataset without duplicating it again,
- keep a clean history of exactly which hyperparameters produced which
  result,
- let someone else run your experiment without reading your code first.

At that point, the thing worth optimizing for is **being able to run
many experiments cheaply and predictably** — not raw simplicity of a
single script. This project's whole structure exists to make that cheap:

> **A new experiment = a new (or edited) file under `configs/`.**
> **A new model = one new file under `modeling/meta_arch/`.**
> **A new dataset = one new file under `data/datasets/`.**
> Nothing else has to change.

This is the same problem Detectron2 (Meta's object detection framework)
solved for detection/segmentation research, where "try a different
backbone/head/dataset without rewriting the training script" is a daily
need. We deliberately borrowed its architecture — registries/catalogs,
a declarative config system, a hook-based training loop, a pluggable
evaluator interface — because those patterns solve exactly this problem,
and they're proven at a much larger scale than we need here. Classification
is simpler than detection, so we only took the pieces that earn their
keep for classification (see [the last section](#why-classification-only)
for what we left out).

---

## The big picture

Here's what actually happens, end to end, when you run
`python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py`:

```
configs/MNIST/mlp_baseline.py            <- you point train_net.py at this
        │
        │  (a plain Python file that imports & composes smaller config
        │   pieces: which model, which data, which optimizer, which
        │   training settings)
        ▼
all_in_one_vision/config/                <- LazyConfig.load() executes that
  (LazyConfig, LazyCall, instantiate)        file and turns it into a config
        │                                    tree; instantiate() later turns
        │                                    parts of that tree into REAL
        │                                    Python objects (a model, an
        │                                    optimizer, a DataLoader...)
        ▼
   ┌────────────┬──────────────┬──────────────┐
   ▼            ▼              ▼              ▼
modeling/    data/          solver/       (train.* plain settings,
meta_arch/   (catalog +     (optimizer       e.g. output_dir,
(the model)   build_loader)  params)          max_epochs, MLflow flag)
   │            │              │                     │
   └────────────┴──────┬───────┴─────────────────────┘
                        ▼
              all_in_one_vision/engine/
              (SimpleTrainer runs the loop;
               Hooks attach checkpointing,
               evaluation, logging, MLflow)
                        │
          ┌─────────────┼──────────────┐
          ▼              ▼              ▼
   evaluation/      checkpoint/     MLflow (external)
   (accuracy)       (save/load)     (experiment tracking)
```

Read that diagram as: **configs describe, `config/` resolves, everything
else does the actual work, `engine/` orchestrates.**

---

## Package-by-package tour

Each of these is a Python package under `all_in_one_vision/`. Click the
package name in your editor to see its own docstrings — this section
gives you the "why", the module docstrings give you the "how".

### `config/` — turning a `.py` file into real objects
The mechanism that makes "new experiment = new config file" possible.
Deep-dived in [`02_config_system_deep_dive.md`](./02_config_system_deep_dive.md).
Short version: config files don't build objects directly — they build
*descriptions* of objects (`LazyCall`), and `instantiate()` turns those
descriptions into the real thing only when training actually starts.

### `data/` — datasets, decoupled from how they're used
- `catalog.py` — a name → dataset-factory registry (`DatasetCatalog`).
  Nothing outside this file needs to know *how* to construct an MNIST
  dataset — it just asks the catalog for `"mnist_train"`.
- `build.py` — `build_loader()`, which turns a catalog name + a
  transform + batch settings into a real `torch.utils.data.DataLoader`.
- `datasets/` — one file per dataset, each registering its dataset(s)
  with the catalog. `mnist.py` is the only one so far.
- `transforms/` — currently just a placeholder; see its own README.

### `modeling/` — the models themselves
- `meta_arch/` — "meta-architecture", Detectron2's term for a
  complete, top-level model (as opposed to a sub-component like a
  backbone). `SimpleMLPClassifier` and `SimpleCNNClassifier` live here.
  Both follow the same **forward contract**, explained below, which is
  what lets `engine/` and `evaluation/` work with either one unchanged.
- `backbone/`, `heads/` — empty placeholder packages. Not used in v1.
  They exist so that if a future model needs a swappable feature
  extractor (e.g. several meta-archs sharing one ResNet backbone), there's
  already an obvious, conventional place to put it — see their READMEs.

### `engine/` — the training loop and its extension points
- `train_loop.py` — `SimpleTrainer`, the actual "for each batch: forward,
  backward, step" loop. It's deliberately generic: it knows nothing about
  MNIST, MLPs, or CNNs.
- `hooks.py` — everything the loop does *besides* the core forward/backward
  step (logging, checkpointing, evaluating, MLflow) is a **hook**, plugged
  in from outside. This is the extension point: adding new training-time
  behavior later means adding a new hook, never editing the loop itself.

### `evaluation/` — measuring how good the model is
`DatasetEvaluator` is an interface (`process()` a batch, `evaluate()` at
the end); `AccuracyEvaluator` is the one implementation we have. Written
as an interface (rather than hardcoding "compute accuracy" into the
training loop) so a future evaluator — e.g. per-class accuracy — can be
swapped in without touching `engine/`.

### `checkpoint/` — saving/loading model weights
A thin subclass of `fvcore`'s `Checkpointer` (the same one Detectron2
uses). We don't reimplement checkpoint saving/loading logic — that's a
solved problem, so we reuse the solution.

### `solver/` — connecting a live model to its optimizer
Small on purpose. See `02_config_system_deep_dive.md` for why building
an optimizer needs a small dance (`get_default_optimizer_params`) that
none of the other components need.

### `utils/` — small, boring, reused-not-reinvented helpers
Logging setup and environment info collection. Nothing
classification-specific lives here.

### `configs/` (project root, not inside `all_in_one_vision/`)
Pure data, not library code — every file here is something you're
expected to read, copy, and edit. See `common/` (shared building blocks)
vs `MNIST/` (a specific experiment that composes those building blocks).

### `tools/train_net.py`
The one script you actually run. On purpose, it contains almost no logic
of its own — it just calls into the packages above, in order. Fully
traced in [`03_training_workflow_walkthrough.md`](./03_training_workflow_walkthrough.md).

---

## Five core concepts, explained from scratch

If you've used Detectron2, scikit-learn's `Pipeline`, or Hydra before,
skip this section — you'll recognize all of it. Otherwise, read this
before diving into any single file; every module docstring in the
codebase assumes you already know these five ideas.

### 1. The Catalog pattern (a name → object registry)

**The problem it solves:** if every piece of code that needs "the MNIST
training set" has to know it's `torchvision.datasets.MNIST(root=...,
train=True, download=True, transform=...)`, then adding a second dataset
means hunting down every place that assumed MNIST specifically.

**The fix:** register a dataset once, under a string name:
```python
DatasetCatalog.register("mnist_train", lambda transform: MNIST(...))
```
...and everywhere else, refer to it only by that name: `"mnist_train"`.
Adding CIFAR-10 later is "write one new file that registers
`cifar10_train`" — nothing that already refers to datasets by name has
to change. See `all_in_one_vision/data/catalog.py`.

### 2. The declarative config pattern (LazyCall + instantiate)

**The problem it solves:** if a config file directly writes
`model = SimpleMLPClassifier(hidden_dim=128)`, that line *runs the
constructor immediately* when the file is imported — before you've had a
chance to override `hidden_dim` from the command line, before you know
if you even want this model or a different one from the same file.

**The fix:** wrap the call so it's recorded, not executed:
```python
model = L(SimpleMLPClassifier)(hidden_dim=128)   # just data, not an object yet
...
model = instantiate(cfg.model)                    # NOW it becomes a real object
```
This is genuinely the single most important idea in the codebase — it's
worth its own document. See
[`02_config_system_deep_dive.md`](./02_config_system_deep_dive.md).

### 3. The forward-contract pattern (how models plug into the trainer)

**The problem it solves:** if the training loop has to know "this model
takes an image and a label and computes cross-entropy loss like *this*",
then every new model needs the loop to be edited too.

**The fix:** every meta-arch agrees to a shared contract:
- in **training mode**, `forward(images, targets)` returns a **dict of
  named losses**, e.g. `{"loss_cls": tensor(...)}`
- in **eval mode**, `forward(images)` returns raw **logits**

`SimpleTrainer` only ever does `loss_dict = model(images, targets)` then
`sum(loss_dict.values()).backward()` — it never needs to know *what's
inside* that dict, or how the model computed it. A future model with two
loss terms (e.g. classification + an auxiliary loss) plugs in with zero
changes to `engine/`. See the docstring at the top of
`modeling/meta_arch/mlp.py` for the full reasoning.

### 4. The hook pattern (extending a loop without editing it)

**The problem it solves:** "print the loss every 20 steps AND save a
checkpoint every epoch AND run evaluation AND log to MLflow" is four
unrelated behaviors. Hardcoding all four into the training loop makes
the loop harder to read and makes "I don't want MLflow this run" awkward.

**The fix:** the loop only calls four generic extension points —
`before_train`, `before_step`, `after_step`, `after_train` — and any
number of independent `Hook` objects can attach to them:
```python
trainer.register_hooks([
    LoggingHook(period=100),
    CheckpointHook(checkpointer, period=iters_per_epoch),
    EvalHook(eval_period=iters_per_epoch, eval_function=do_eval),
    MLflowHook(...) if cfg.train.mlflow.enabled else None,
])
```
Each hook does exactly one thing and knows nothing about the others. See
`all_in_one_vision/engine/hooks.py`.

### 5. The evaluator pattern (measuring performance as a swappable piece)

**The problem it solves:** "loop over the test set and compute accuracy"
is logic that shouldn't live inside the training loop (it's a different
concern), but also shouldn't be duplicated every time you want a
different metric.

**The fix:** `DatasetEvaluator` is a two-method interface —
`process(targets, outputs)` per batch, `evaluate()` once at the end — and
`inference_on_dataset(model, loader, evaluator)` is the one generic
function that drives any evaluator through a full pass over a dataset.
See `all_in_one_vision/evaluation/evaluator.py`.

---

## Why classification-only

Detectron2 itself contains a lot of machinery this project intentionally
does **not** copy, because it exists to solve *detection*-specific
problems that classification doesn't have:

| Detectron2 has... | ...because detection needs... | Do we need it? |
|---|---|---|
| `structures/` (Boxes, Instances, ImageList) | bounding boxes, variable-size outputs per image | No — a classification label is just an int |
| `modeling/roi_heads/`, `anchor_generator.py`, `proposal_generator/` | region proposals, ROI pooling | No — no regions to propose |
| `modeling/backbone/fpn.py` and friends | multi-scale feature maps for objects of different sizes | No — not yet; see `backbone/README.md` for when this *would* become relevant |
| `evaluation/coco_evaluation.py`, `pascal_voc_evaluation.py` | mAP-style detection metrics | No — accuracy is enough for classification |

If a future version of this project needs true multi-scale feature
extraction (e.g. for robotics vision tasks with varying object sizes),
the `backbone/` and `heads/` placeholder packages are exactly where that
would go — but nothing detection-specific (boxes, proposals, ROI
pooling) is planned or assumed anywhere in this codebase.

➡️ Next: [`02_config_system_deep_dive.md`](./02_config_system_deep_dive.md)
