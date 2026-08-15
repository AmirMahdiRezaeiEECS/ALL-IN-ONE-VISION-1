# The Config System, Deep Dive

This document explains `all_in_one_vision/config/` completely — by the
end, you should be able to predict exactly what any config file in this
project turns into, without running it.

There are three pieces, in three files, and they're deliberately kept
separate because they do three different jobs:

| File | Job | Analogy |
|---|---|---|
| `lazy_call.py` (`LazyCall`, imported as `L`) | Record "build this, with these arguments" as data | Writing a recipe |
| `instantiate.py` (`instantiate()`) | Turn a recorded recipe into a real object | Cooking the recipe |
| `lazy.py` (`LazyConfig`) | Load a `.py` config file and apply CLI overrides | Grabbing the recipe book and swapping an ingredient |

If you already read the "declarative config pattern" section in
`01_architecture_and_concepts.md`, this document is the detailed version
of that idea.

## Table of contents
- [The problem: why not just construct objects directly?](#the-problem)
- [LazyCall: building a recipe, not an object](#lazycall)
- [instantiate(): cooking the recipe](#instantiate)
- [LazyConfig.load(): turning a .py file into a config tree](#lazyconfigload)
- [CLI overrides: how `train.max_epochs=10` works](#cli-overrides)
- [Worked example: fully tracing `mlp_baseline.py`](#worked-example)
- [The one tricky case: the optimizer](#the-optimizer-special-case)

---

## The problem

Suppose `configs/MNIST/mlp_baseline.py` just did this:

```python
from all_in_one_vision.modeling.meta_arch import SimpleMLPClassifier
model = SimpleMLPClassifier(hidden_dim=128, num_classes=10)
```

The instant Python imports this file, `SimpleMLPClassifier(...)` runs —
a real `nn.Module` gets constructed, weights get randomly initialized,
everything. That happens the moment `train_net.py` does
`LazyConfig.load(...)`, **before** any command-line overrides have been
applied, and before we've decided whether we even want this model or a
different one referenced later in the same file.

We want configs to be **inert data** until the exact moment we're ready
to build things — so that `train.max_epochs=10` typed on the command
line can change the config *before* anything gets built, and so a config
can be printed/inspected/logged (e.g. to MLflow) without side effects.

---

## LazyCall

`LazyCall` (imported everywhere as `L`, matching Detectron2's convention)
wraps a class or function so that *calling it* produces a small
dictionary describing the call, instead of making the call:

```python
from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.modeling.meta_arch import SimpleMLPClassifier

model = L(SimpleMLPClassifier)(hidden_dim=128, num_classes=10)
```

After this line, `model` is **not** a `SimpleMLPClassifier`. Print it and
you'd see something like:

```python
{
    "_target_": "all_in_one_vision.modeling.meta_arch.mlp.SimpleMLPClassifier",
    "hidden_dim": 128,
    "num_classes": 10,
}
```

Two things worth noticing:
- `_target_` is a **string** (the class's full import path), not the
  class itself. This keeps the recipe as plain, serializable data.
- The dict is actually an OmegaConf `DictConfig`, not a plain `dict` —
  that's what gives us dotted-key access (`cfg.model.hidden_dim`) and the
  CLI-override machinery described below. You can treat it like a dict
  for almost all purposes.

**Recipes nest.** A recipe's keyword arguments can themselves be
recipes — this is how the whole config tree (model → optimizer →
dataloader → ...) stays inert until `instantiate()` is called on it.

---

## instantiate()

`instantiate()` is the reverse operation: walk a config tree, and
wherever there's a `"_target_"` key, resolve that string back into the
real class/function, recursively instantiate its arguments first, then
actually call it.

```python
from all_in_one_vision.config import instantiate

real_model = instantiate(model)   # NOW SimpleMLPClassifier(hidden_dim=128, ...) actually runs
assert isinstance(real_model, SimpleMLPClassifier)
```

The recursion matters: if one of the keyword arguments is itself a
`LazyCall` recipe (e.g. an optimizer's `params=` argument being another
recipe — see [below](#the-optimizer-special-case)), `instantiate()`
resolves that inner recipe *first*, then passes the real, already-built
result as the argument. This is what lets arbitrarily deep object graphs
(a trainer containing a model containing layers...) be described as pure
data and built in one call.

Anything that isn't a recipe (plain numbers, strings, lists, or objects
that are already "real", like a loaded tensor) passes through
`instantiate()` unchanged. This means you can freely mix declarative
recipes and plain Python values in the same config — nothing requires
*everything* to go through `L(...)`.

---

## LazyConfig.load()

A single `L(...)` recipe is just one object. A full experiment needs
several: a model, a dataloader, an optimizer, and some plain training
settings (output directory, number of epochs...). `LazyConfig.load()`
is what turns an entire `.py` file into one combined config tree.

Concretely, `configs/MNIST/mlp_baseline.py` looks like this:

```python
from configs.common.models.mlp import model
from configs.common.data.mnist import dataloader
from configs.common.optim import optimizer
from configs.common.train import train

train.output_dir = "./output/mnist_mlp_v1"
train.max_epochs = 5
```

`LazyConfig.load("configs/MNIST/mlp_baseline.py")` does roughly this:
1. `exec()`s the file's source code in an isolated namespace (so running
   it doesn't pollute your actual Python session).
2. After execution, that namespace contains variables `model`,
   `dataloader`, `optimizer`, `train` (imported from the `common/` files,
   then the last two lines mutate `train` in place).
3. It collects every "public" top-level variable (skipping anything
   starting with `_`, and skipping imported modules themselves — we only
   want the `model`/`dataloader`/`optimizer`/`train` *values*, not the
   `configs.common.models.mlp` module they came from) into one
   `OmegaConf` tree:

```python
cfg = OmegaConf.create({
    "model": model,          # the L(...) recipe from step 1 above
    "dataloader": dataloader,
    "optimizer": optimizer,
    "train": train,
})
```

Now `cfg.model`, `cfg.dataloader.train`, `cfg.optimizer.lr`, and
`cfg.train.max_epochs` are all accessible with dotted syntax on one
object, and nothing has been *built* yet — it's all still inert data.

> **Why `configs/` is a real Python package.** Notice
> `from configs.common.models.mlp import model` is an ordinary, absolute
> Python import. Upstream Detectron2 patches Python's import system to
> allow *relative* imports between config files that technically aren't
> a package. We chose the simpler route: `configs/` has an `__init__.py`
> at every level, making it a real, importable package, so plain absolute
> imports just work with zero custom machinery. See
> `07_faq_and_design_deviations.md` for the full reasoning.

---

## CLI overrides

`train_net.py` accepts overrides like:

```bash
python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py \
    train.max_epochs=10 optimizer.lr=0.005
```

`LazyConfig.apply_overrides(cfg, ["train.max_epochs=10", "optimizer.lr=0.005"])`
handles this by, for each `key=value` string:
1. Splitting on the first `=`.
2. Parsing `value` with YAML (`yaml.safe_load`), so `"10"` → the int
   `10`, `"0.005"` → the float `0.005`, `"true"` → the bool `True`,
   without you needing to worry about types.
3. Calling `OmegaConf.update(cfg, "train.max_epochs", 10, merge=True)`,
   which walks the dotted path and sets that exact field.

Because this happens *after* `LazyConfig.load()` but *before*
`instantiate()` is ever called on anything, overriding a value is just
editing plain data — there's no already-built object anywhere yet to
get out of sync.

---

## Worked example

Let's trace the entire `mlp_baseline.py` pipeline, step by step, matching
exactly what `tools/train_net.py` does.

**1. Load the config file:**
```python
cfg = LazyConfig.load("configs/MNIST/mlp_baseline.py")
```
`cfg` is now a tree of recipes + plain settings:
```
cfg.model        = {"_target_": "....SimpleMLPClassifier", "hidden_dim": 128, ...}
cfg.dataloader.train = {"_target_": "....build_loader", "dataset_name": "mnist_train", ...}
cfg.dataloader.test  = {"_target_": "....build_loader", "dataset_name": "mnist_test", ...}
cfg.optimizer     = {"_target_": "torch.optim.SGD", "params": {"_target_": "....get_default_optimizer_params", "model": None}, "lr": 0.01, "momentum": 0.9}
cfg.train         = {"output_dir": "./output/mnist_mlp_v1", "max_epochs": 5, ...}
```

**2. Apply any CLI overrides** (none in this example — skip).

**3. Build the model:**
```python
model = instantiate(cfg.model)
# -> resolves "....SimpleMLPClassifier", calls
#    SimpleMLPClassifier(in_features=784, hidden_dim=128, num_classes=10)
# -> a real nn.Module, randomly initialized
```

**4. Build the data loaders:**
```python
train_loader = instantiate(cfg.dataloader.train)
# -> resolves "....build_loader", recursively instantiates its `transform`
#    argument (itself a recipe: L(T.Compose)(transforms=[L(T.ToTensor)(), ...])),
#    then calls build_loader(dataset_name="mnist_train", transform=<real Compose>, ...)
# -> a real torch.utils.data.DataLoader
```

**5. Build the optimizer** — this one needs one extra step. See the next
section.

**6. Everything from here on (`SimpleTrainer`, hooks, evaluation) works
with real, already-built objects** — `instantiate()`'s job is done.

---

## The optimizer special case

Every other object in this project can be described purely from config —
but an optimizer's constructor needs `model.parameters()`, and at
config-*authoring* time (when you're writing the `.py` file), no model
object exists yet. `configs/common/optim.py` handles this by leaving a
placeholder:

```python
optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(model=None),   # <- placeholder!
    lr=0.01,
    momentum=0.9,
)
```

`tools/train_net.py` fills the placeholder in with the real model —
which by this point in the script has already been built in step 3
above — right before instantiating the optimizer:

```python
model = instantiate(cfg.model)          # step 3, already have this
...
cfg.optimizer.params.model = model      # inject the REAL model object into the recipe
optimizer = instantiate(cfg.optimizer)  # NOW get_default_optimizer_params(model=<real model>)
                                          #     runs, returning model.parameters(),
                                          # THEN torch.optim.SGD(params=<that>, lr=0.01, ...) runs
```

This is the only place in the whole codebase where a config value is set
to an already-built Python object rather than a plain value or another
recipe — and it's exactly why `LazyCall.__call__` sets
`flags={"allow_objects": True}` on the `DictConfig` it creates (see the
docstring in `lazy_call.py`): without that flag, OmegaConf would refuse
to store a live model object inside a config tree.

If you ever add a new component whose constructor needs a real,
already-built object as an argument, this is the pattern to copy:
1. Leave a `None` placeholder for that argument in the config.
2. Right before `instantiate()`-ing that piece, inject the real object
   with `cfg.<path>.<field> = real_object`.

➡️ Next: [`03_training_workflow_walkthrough.md`](./03_training_workflow_walkthrough.md)
