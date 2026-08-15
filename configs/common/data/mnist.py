"""
Data config: MNIST
====================
This building-block config describes:
  1. `transform`  -- how a raw MNIST image gets turned into a tensor the
                       model can consume (ToTensor + Normalize).
  2. `dataloader.train` / `dataloader.test` -- ready-to-instantiate
                       recipes for the training and test DataLoaders.

WHERE THE ACTUAL DATA COMES FROM: notice this file never mentions the
MNIST class or a file path directly -- it only refers to the dataset by
NAME ("mnist_train" / "mnist_test"), which is looked up in the
DatasetCatalog at build time. The actual registration (which torchvision
class to use, where files get downloaded to) lives in
all_in_one_vision/data/datasets/mnist.py. This split -- "which dataset"
here, "how to construct it" there -- is what lets a new dataset be added
without ever touching an experiment config's structure. See
docs/01_architecture_and_concepts.md ("The Catalog pattern") and
docs/04_extending_the_project.md ("Adding a new dataset").

WHY `OmegaConf.create()` FOR `dataloader`: `dataloader` needs to hold TWO
separate recipes (.train and .test) as named sub-fields, so we build an
empty config node first and then assign each one. `transform` is shared
between them (same normalization for train and test).

THE NORMALIZE VALUES (mean=0.1307, std=0.3081) are MNIST's actual,
well-known per-channel pixel statistics computed over the training set --
not arbitrary numbers. Every MNIST tutorial you'll find uses the same
constants; a different dataset needs its own (see
configs/common/data/... for a Fashion-MNIST example in the extending guide).
"""
from omegaconf import OmegaConf
from torchvision import transforms as T

from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.data import build_loader

transform = L(T.Compose)(
    transforms=[
        L(T.ToTensor)(),                                    # PIL image -> float tensor in [0, 1]
        L(T.Normalize)(mean=[0.1307], std=[0.3081]),          # standardize using MNIST's own stats
    ]
)

dataloader = OmegaConf.create()
dataloader.train = L(build_loader)(
    dataset_name="mnist_train",
    transform=transform,
    batch_size=64,
    shuffle=True,     # shuffle every epoch during training, to avoid batch-order bias
    num_workers=0,     # 0 = load data in the main process; raise this for faster loading
                        # on a machine with multiple CPU cores available.
)
dataloader.test = L(build_loader)(
    dataset_name="mnist_test",
    transform=transform,
    batch_size=1000,   # larger batch for eval is fine -- no gradients are computed
    shuffle=False,      # no need to shuffle when just measuring accuracy
    num_workers=0,
)
