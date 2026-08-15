# FAQ & Design Deviations from Upstream Detectron2

This project deliberately imitates Detectron2's architecture, but isn't
a line-for-line port — some of Detectron2's real machinery is more than
a classification MVP needs, and a few things were simplified on purpose.
This document is the single place all of those decisions are collected
and explained, so nothing feels like an unexplained inconsistency if
you've read (or later read) Detectron2's own source.

## Table of contents
- [Why LazyConfig instead of the older YACS CfgNode system?](#why-lazyconfig)
- [Why absolute imports between config files, not relative?](#why-absolute-imports)
- [Why no fvcore Registry for meta-archs?](#why-no-registry-for-meta-archs)
- [Why iteration-based training instead of epoch-based?](#why-iteration-based)
- [Why is `get_default_optimizer_params` so minimal?](#why-minimal-optimizer-params)
- [Why do `backbone/` and `heads/` exist if they're empty?](#why-empty-backbone-heads)
- [Why torchmetrics for accuracy instead of hand-written code?](#why-torchmetrics)
- [Why fvcore's Checkpointer instead of `torch.save`/`torch.load` directly?](#why-fvcore-checkpointer)
- [I don't know Detectron2 — do I need to, to use this project?](#do-i-need-detectron2-knowledge)

---

### Why LazyConfig instead of the older YACS CfgNode system? {#why-lazyconfig}

Detectron2 actually ships *two* config systems: an older YACS-based
`CfgNode` system (simpler, YAML-based) and a newer, Python-based
`LazyConfig` system. We chose LazyConfig for this project — it was an
explicit decision made and confirmed early in this project's design
discussion, favoring the more flexible, code-native approach that
matches where Detectron2 itself has been moving, even though it's
slightly more machinery to understand upfront (see
`02_config_system_deep_dive.md`).

### Why absolute imports between config files, not relative? {#why-absolute-imports}

Upstream Detectron2's `LazyConfig.load()` patches Python's import
machinery (`builtins.__import__`) so that config files can use relative
imports (`from .common.optim import optimizer`) despite not technically
being part of an installed package. That's clever but adds real
complexity for a benefit this project doesn't need yet.

Here, `configs/` is simply a real, ordinary Python package — every
directory has an `__init__.py`. Config files use plain, ordinary
absolute imports:
```python
from configs.common.models.mlp import model
```
Every other part of the LazyConfig philosophy (declarative `L(...)`
recipes, `instantiate()`, dotted-key CLI overrides) is unchanged. If
config files ever need to live *outside* this package (e.g.
user-supplied configs elsewhere on disk, not checked into this repo),
that's the point to revisit this and consider the upstream-style import
patching.

### Why no fvcore Registry for meta-archs? {#why-no-registry-for-meta-archs}

Detectron2 does use an `fvcore.common.registry.Registry` for some
pluggable components (e.g. `ROI_HEADS_REGISTRY`), where a config
selects an implementation **by string name**
(`cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"`), and a `build_*(cfg)`
function looks that name up in the registry.

With LazyConfig, the idiomatic pattern is usually simpler: reference the
class **directly** in the config file (`model = L(SimpleMLPClassifier)(...)`)
rather than by a string name looked up in a registry — Detectron2's own
LazyConfig-based configs (e.g. its ViTDet configs) do exactly this for
the top-level model. We follow that same convention for `meta_arch/`
here. `fvcore`'s `Registry` is still a project dependency and stays in
reserve for `backbone/`, if/when a component genuinely needs to be
swappable *by name* rather than by direct reference (see the next
question).

### Why iteration-based training instead of epoch-based? {#why-iteration-based}

The original standalone v1 MLP script trained "for 5 epochs" directly.
`SimpleTrainer` (in `engine/train_loop.py`) instead trains for a number
of **iterations** (one iteration = one batch), matching Detectron2's own
`SimpleTrainer` exactly. `tools/train_net.py` converts your config's
`train.max_epochs` into `max_iter` once, near the top (see
`03_training_workflow_walkthrough.md`, step 6), so you can still *think*
and *write configs* in epochs — only the loop's internals count
iterations. This was a deliberate choice to keep the loop itself, and
every hook's "every N iterations" logic, uniform and simple, exactly as
in upstream Detectron2.

### Why is `get_default_optimizer_params` so minimal? {#why-minimal-optimizer-params}

Detectron2's real version supports per-parameter-group overrides (e.g. a
different learning rate or weight decay for bias parameters vs. weight
parameters, or for backbone parameters vs. head parameters). Our version
(`solver/build.py`) is just `model.parameters()` — a classification MLP/
CNN MVP doesn't need per-group tuning yet. The function is intentionally
kept as the single, obvious place to add that behavior later, without
needing to touch `configs/common/optim.py` or `tools/train_net.py` — only
this one function's internals would change.

### Why do `backbone/` and `heads/` exist if they're empty? {#why-empty-backbone-heads}

Both `SimpleMLPClassifier` and `SimpleCNNClassifier` are small enough to
be complete, self-contained meta-archs (matching Detectron2's simplest
meta-archs, which also don't split out a separate backbone). These
placeholder packages exist so that *if* a future model needs a genuinely
swappable feature extractor — e.g. several different meta-archs all
built on top of one shared, interchangeable backbone (a ResNet, for
instance) — there's already an established, conventional location for
it, and adding it won't require restructuring `modeling/`. Until that's
actually needed, they stay empty rather than containing speculative,
unused code. See their own `README.md` files for the same explanation
in context.

### Why torchmetrics for accuracy instead of hand-written code? {#why-torchmetrics}

Reuse-first: `torchmetrics.Accuracy` is a tested, maintained
implementation that correctly handles edge cases (e.g. different
averaging strategies for imbalanced classes) that a hand-rolled
`(preds == targets).float().mean()` would get subtly wrong in some
configurations. `evaluation/classification_evaluation.py::AccuracyEvaluator`
is a thin adapter, not a reimplementation.

### Why fvcore's Checkpointer instead of `torch.save`/`torch.load` directly? {#why-fvcore-checkpointer}

Same reuse-first reasoning: `fvcore.common.checkpoint.Checkpointer`
already handles saving/loading model *and* optimizer state together,
finding "the latest checkpoint" in a directory, and other small but
easy-to-get-wrong details — and it's the exact class Detectron2 itself
uses, so staying consistent with it costs nothing extra.

### I don't know Detectron2 — do I need to, to use this project? {#do-i-need-detectron2-knowledge}

No. Everything you need to understand *this* project is in this `docs/`
folder — start at `00_start_here.md`. Detectron2 is mentioned throughout
only to explain *why* certain patterns exist (they're proven at a larger
scale in a well-known open-source project), not because using this
project requires having used Detectron2 first.
