# Extending the Project: Adding a Dataset or Model

This is a practical, copy-paste-and-adapt guide. If you've read
`01_architecture_and_concepts.md` and `02_config_system_deep_dive.md`,
everything here should feel like it's just applying what you already
know — this doc exists so you don't have to re-derive it every time.

## Table of contents
- [Adding a new dataset](#adding-a-new-dataset)
- [Adding a new model](#adding-a-new-model)
- [Adding a new hyperparameter / config field](#adding-a-new-hyperparameter)
- [A note on what NOT to add](#a-note-on-what-not-to-add)

---

## Adding a new dataset

Worked example: adding **Fashion-MNIST** (same shape as MNIST, different
images, also built into `torchvision`, so this is close to the simplest
possible new dataset).

### Step 1 — register it in the `DatasetCatalog`

Create `all_in_one_vision/data/datasets/fashion_mnist.py`:

```python
"""
Fashion-MNIST dataset registration. See mnist.py for the pattern this follows.
"""
from torchvision.datasets import FashionMNIST

from ..catalog import DatasetCatalog, MetadataCatalog

_DEFAULT_ROOT = "./datasets/fashion_mnist"

_METADATA = {
    "num_classes": 10,
    "class_names": [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
    ],
}


def _make(split: str, root: str = _DEFAULT_ROOT):
    is_train = split == "train"

    def factory(transform):
        return FashionMNIST(root=root, train=is_train, download=True, transform=transform)

    return factory


DatasetCatalog.register("fashion_mnist_train", _make("train"))
DatasetCatalog.register("fashion_mnist_test", _make("test"))
MetadataCatalog.register("fashion_mnist_train", dict(_METADATA))
MetadataCatalog.register("fashion_mnist_test", dict(_METADATA))
```

### Step 2 — register the import so it actually runs

Edit `all_in_one_vision/data/datasets/__init__.py`:

```python
from . import mnist            # noqa: F401
from . import fashion_mnist    # noqa: F401  <- add this line

__all__ = ["mnist", "fashion_mnist"]
```

(Registration happens as an import *side effect* — importing
`all_in_one_vision.data` registers every dataset listed here. This is
why the import has to be added, even though nothing calls a function
directly.)

### Step 3 — write the config building block

Create `configs/common/data/fashion_mnist.py` (nearly identical to
`configs/common/data/mnist.py` — same image shape, same transform is
fine as a starting point):

```python
from omegaconf import OmegaConf
from torchvision import transforms as T

from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.data import build_loader

transform = L(T.Compose)(
    transforms=[
        L(T.ToTensor)(),
        L(T.Normalize)(mean=[0.2860], std=[0.3530]),  # Fashion-MNIST's own stats
    ]
)

dataloader = OmegaConf.create()
dataloader.train = L(build_loader)(
    dataset_name="fashion_mnist_train",
    transform=transform,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
dataloader.test = L(build_loader)(
    dataset_name="fashion_mnist_test",
    transform=transform,
    batch_size=1000,
    shuffle=False,
    num_workers=0,
)
```

### Step 4 — write an experiment config that uses it

Create `configs/Fashion-MNIST/mlp_baseline.py` (mkdir the folder first,
and add an `__init__.py` so it's importable, matching `configs/MNIST/`):

```python
from configs.common.models.mlp import model
from configs.common.data.fashion_mnist import dataloader
from configs.common.optim import optimizer
from configs.common.train import train

train.output_dir = "./output/fashion_mnist_mlp_v1"
train.max_epochs = 5
train.mlflow.experiment_name = "all-in-one-vision-fashion-mnist"
```

### Step 5 — run it

```bash
python tools/train_net.py --config-file configs/Fashion-MNIST/mlp_baseline.py
```

That's it. **Nothing in `all_in_one_vision/engine/`, `evaluation/`, or
`tools/train_net.py` needed to change.** This is the concrete payoff of
the catalog pattern from `01_architecture_and_concepts.md`.

---

## Adding a new model

Worked example: adding a slightly deeper MLP (`DeepMLPClassifier`) — any
new model follows this exact shape, whether it's a small variant like
this or something structurally different.

### Step 1 — write the model, following the forward contract

Create `all_in_one_vision/modeling/meta_arch/deep_mlp.py`:

```python
"""
DeepMLPClassifier: a 3-hidden-layer variant of SimpleMLPClassifier.
Follows the same train/eval forward contract -- see mlp.py for why that
contract exists and what it requires.
"""
import torch.nn as nn
import torch.nn.functional as F


class DeepMLPClassifier(nn.Module):
    def __init__(self, in_features=28 * 28, hidden_dims=(256, 128, 64), num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        dims = [in_features, *hidden_dims]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.out = nn.Linear(dims[-1], num_classes)

    def forward(self, images, targets=None):
        x = self.flatten(images)
        for layer in self.layers:
            x = F.relu(layer(x))
        logits = self.out(x)

        if self.training:
            assert targets is not None, "targets are required in training mode"
            loss = F.cross_entropy(logits, targets)
            return {"loss_cls": loss}

        return logits
```

**The two things that make a class usable as a meta-arch:**
1. In training mode (`self.training == True`), `forward(images, targets)`
   returns a `dict` of named loss tensors (each a scalar).
2. In eval mode, `forward(images)` returns raw logits, shape
   `(batch_size, num_classes)`.

That's the entire requirement — no base class to inherit from, no
registry to update by hand.

### Step 2 — export it

Edit `all_in_one_vision/modeling/meta_arch/__init__.py`:

```python
from .mlp import SimpleMLPClassifier
from .cnn import SimpleCNNClassifier
from .deep_mlp import DeepMLPClassifier   # <- add this line

__all__ = ["SimpleMLPClassifier", "SimpleCNNClassifier", "DeepMLPClassifier"]
```

### Step 3 — write the config building block

Create `configs/common/models/deep_mlp.py`:

```python
from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.modeling.meta_arch import DeepMLPClassifier

model = L(DeepMLPClassifier)(
    in_features=28 * 28,
    hidden_dims=(256, 128, 64),
    num_classes=10,
)
```

### Step 4 — write an experiment config that uses it

```python
# configs/MNIST/deep_mlp_baseline.py
from configs.common.models.deep_mlp import model
from configs.common.data.mnist import dataloader
from configs.common.optim import optimizer
from configs.common.train import train

train.output_dir = "./output/mnist_deep_mlp_v1"
train.max_epochs = 5
```

### Step 5 — run it

```bash
python tools/train_net.py --config-file configs/MNIST/deep_mlp_baseline.py
```

Again: **`engine/`, `evaluation/`, `tools/train_net.py` unchanged.** This
is the forward-contract pattern paying off — see
`01_architecture_and_concepts.md` if you want the "why" again.

---

## Adding a new hyperparameter / config field

If you just want to expose something as tunable without writing new
code — e.g. making `SimpleCNNClassifier`'s number of channels
configurable, which it already is — you likely don't need to touch
anything except the config file. Every constructor argument on a
registered model/dataset/etc. is automatically overridable, either by
editing the config file directly, or from the CLI:

```bash
python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py \
    model.hidden_dim=256
```

You only need code changes (Step 1 in the sections above) when the thing
you want to change genuinely isn't a constructor argument yet — e.g. the
number of hidden layers in `SimpleMLPClassifier`, which is currently
fixed at one.

---

## A note on what NOT to add

Per this project's scope (see `01_architecture_and_concepts.md`'s
["why classification-only"](./01_architecture_and_concepts.md#why-classification-only)
section): resist the urge to add detection/segmentation-specific
abstractions (bounding boxes, anchors, ROI pooling, region proposals)
even if you've seen them in Detectron2 and they seem "more complete".
They solve problems classification doesn't have. If a genuine need for
them arises, that's a deliberate, discussed architectural decision — not
a routine addition.

➡️ Next: [`05_mlflow_and_experiment_tracking.md`](./05_mlflow_and_experiment_tracking.md)
