"""
MNIST dataset registration.

REUSE-FIRST: we don't reimplement MNIST parsing/downloading at all --
torchvision.datasets.MNIST already does that correctly. This module's only
job is to register two named entries in the DatasetCatalog:
"mnist_train" and "mnist_test", each a factory that accepts a `transform`
and returns a ready-to-use torchvision MNIST dataset.

Importing this module registers the datasets as a SIDE EFFECT (matching
Detectron2's builtin.py convention: registration happens on import, not by
calling a function explicitly from training code). `data/datasets/__init__.py`
imports this module so simply importing `all_in_one_vision.data` is enough
to make "mnist_train" / "mnist_test" available.
"""
from torchvision.datasets import MNIST

from ..catalog import DatasetCatalog, MetadataCatalog

_DEFAULT_ROOT = "./datasets/mnist"

_MNIST_METADATA = {
    "num_classes": 10,
    "class_names": [str(i) for i in range(10)],
}


def _make_mnist(split: str, root: str = _DEFAULT_ROOT):
    is_train = split == "train"

    def factory(transform):
        return MNIST(root=root, train=is_train, download=True, transform=transform)

    return factory


DatasetCatalog.register("mnist_train", _make_mnist("train"))
DatasetCatalog.register("mnist_test", _make_mnist("test"))
MetadataCatalog.register("mnist_train", dict(_MNIST_METADATA))
MetadataCatalog.register("mnist_test", dict(_MNIST_METADATA))
