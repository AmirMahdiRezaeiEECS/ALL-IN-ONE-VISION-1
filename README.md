# ALL_IN_ONE_VISION_1

Classification-focused computer vision project, architected to closely
follow **Detectron2's** structure and philosophy (a catalog pattern for
datasets, a LazyConfig-style declarative config system, a hook-based
training engine, a DatasetEvaluator interface), adapted for
classification.

**Scope note:** this is explicitly classification-only. Detection/
segmentation-specific abstractions (boxes, ROI heads, anchors,
proposal generators, etc.) are intentionally NOT part of this project.

## 📚 Documentation — start here

This README is a quick reference. For a full, guided learning path —
assuming zero prior knowledge of this project — start at
**[`docs/00_start_here.md`](docs/00_start_here.md)**:

| Doc | Covers |
|---|---|
| [`00_start_here.md`](docs/00_start_here.md) | Orientation; a working example in 2 minutes |
| [`01_architecture_and_concepts.md`](docs/01_architecture_and_concepts.md) | The big picture: what each package does and *why*; five core concepts explained from scratch |
| [`02_config_system_deep_dive.md`](docs/02_config_system_deep_dive.md) | Exactly how `LazyCall`/`instantiate`/`LazyConfig` work, with a fully worked trace |
| [`03_training_workflow_walkthrough.md`](docs/03_training_workflow_walkthrough.md) | A line-by-line trace of `tools/train_net.py` |
| [`04_extending_the_project.md`](docs/04_extending_the_project.md) | Copy-paste guide: adding a new dataset or model |
| [`05_mlflow_and_experiment_tracking.md`](docs/05_mlflow_and_experiment_tracking.md) | How MLflow is wired in, and how to extend it |
| [`06_testing_guide.md`](docs/06_testing_guide.md) | What the test suite proves and how to extend it |
| [`07_faq_and_design_deviations.md`](docs/07_faq_and_design_deviations.md) | Every deliberate deviation from upstream Detectron2, explained |

## v1 — status

- Dataset: MNIST
- Models: `SimpleMLPClassifier` (ported from the original standalone v1
  script) and `SimpleCNNClassifier` (new)
- Config-driven: swapping model/dataset/hyperparameters requires only a
  new or edited config file under `configs/`, never a source change
- MLflow integrated as a training hook (`engine/hooks.py::MLflowHook`),
  toggled via `train.mlflow.enabled` in config
- `backbone/` and `heads/` exist as placeholder packages (see their
  READMEs) — not used yet, reserved for when a model needs a swappable
  feature extractor

## Project layout

```
all_in_one_vision/     # the library
    config/              # LazyCall, instantiate(), LazyConfig (load .py configs)
    data/                # DatasetCatalog/MetadataCatalog, build_loader, datasets/, transforms/
    modeling/
        meta_arch/          # SimpleMLPClassifier, SimpleCNNClassifier
        backbone/            # placeholder (reserved)
        heads/                # placeholder (reserved)
    solver/               # get_default_optimizer_params
    engine/               # TrainerBase/SimpleTrainer + hooks (Logging/Checkpoint/Eval/MLflow)
    evaluation/            # DatasetEvaluator interface, AccuracyEvaluator
    checkpoint/             # Checkpointer (subclasses fvcore's)
    utils/                   # logger, collect_env

configs/                # pure data — one file per experiment/component
    common/               # shared building blocks (models/, data/, optim.py, train.py)
    MNIST/                 # mlp_baseline.py, cnn_baseline.py

tools/
    train_net.py            # the one CLI entrypoint

tests/                   # pytest suite (config/instantiate mechanics, catalog,
                           # meta_arch forward contract, full pipeline smoke test)
```

## Running an experiment

```bash
# from the project root, with the anaconda env active
pip install -r requirements.txt

python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py
python tools/train_net.py --config-file configs/MNIST/cnn_baseline.py

# override anything from the CLI without editing files:
python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py \
    train.max_epochs=10 optimizer.lr=0.005
```

MLflow runs are written to the default local tracking store. View them with:

```bash
mlflow ui
```

> Note: recent MLflow versions (3.x) put the bare local filesystem store
> ("./mlruns") into maintenance mode and will raise unless you either use
> a database backend (`mlflow.set_tracking_uri("sqlite:///mlflow.db")`)
> or set `MLFLOW_ALLOW_FILE_STORE=true`. This isn't specific to this
> project — worth knowing before your first run.

## Running tests

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

All 11 tests pass as of this version, including a full
load-config → instantiate → train → evaluate → checkpoint smoke test that
runs against a synthetic (network-free) dataset standing in for MNIST.

## Adding a new dataset

1. Write `all_in_one_vision/data/datasets/<name>.py`, registering
   `"<name>_train"` / `"<name>_test"` in `DatasetCatalog` (see `mnist.py`
   for the pattern).
2. Add the import to `all_in_one_vision/data/datasets/__init__.py`.
3. Write `configs/common/data/<name>.py` (transform + `dataloader.train`/`.test`).
4. Reference it from a new `configs/<Dataset>/<experiment>.py`.

No changes to `engine/`, `evaluation/`, or existing configs required.

## Adding a new model

1. Write `all_in_one_vision/modeling/meta_arch/<name>.py`, an `nn.Module`
   following the training/eval forward contract documented in `mlp.py`
   (returns a loss dict in train mode, raw logits in eval mode).
2. Export it from `meta_arch/__init__.py`.
3. Write `configs/common/models/<name>.py` wrapping it in `L(...)`.
4. Reference it from an experiment config.

No changes to `engine/`, `evaluation/`, or `tools/train_net.py` required.

## Known, deliberate deviations from upstream Detectron2

- **Config file imports are absolute, not relative.** Upstream Detectron2
  patches Python's import machinery so config files can use relative
  imports (`from .common.optim import optimizer`) despite not being a
  real installed package. We instead made `configs/` an ordinary Python
  package and use plain absolute imports
  (`from configs.common.optim import optimizer`). Same LazyConfig
  philosophy (declarative recipes, `instantiate()`, dotted CLI
  overrides), simpler import mechanism. Flagged during the original
  design discussion; revisit if config files ever need to live outside
  the package.
- **Iteration-based training, not epoch-based**, matching Detectron2's
  convention (`train.max_epochs` in config is converted to `max_iter`
  once, at the top of `tools/train_net.py`).
- **`get_default_optimizer_params`** (`solver/build.py`) is intentionally
  minimal (`model.parameters()`) — no per-parameter-group overrides yet.
  This is the obvious place to extend if/when needed.
