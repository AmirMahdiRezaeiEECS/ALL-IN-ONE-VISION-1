"""
CIFAR-10 dataset registration.

REUSE-FIRST, same pattern as mnist.py: torchvision.datasets.CIFAR10
already implements correct downloading and parsing. This module's only
job is to register two named entries in the DatasetCatalog:
"cifar10_train" and "cifar10_test", each a factory that accepts a
`transform` and returns a ready-to-use torchvision CIFAR10 dataset.

Registration happens as an import side effect, matching mnist.py's
convention -- see data/datasets/__init__.py.
"""
from torchvision.datasets import CIFAR10

from ..catalog import DatasetCatalog, MetadataCatalog

_DEFAULT_ROOT = "./datasets/cifar10"

_CIFAR10_METADATA = {
    "num_classes": 10,
    "class_names": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ],
}


def _make_cifar10(split: str, root: str = _DEFAULT_ROOT):
    is_train = split == "train"

    def factory(transform):
        return CIFAR10(root=root, train=is_train, download=True, transform=transform)

    return factory


DatasetCatalog.register("cifar10_train", _make_cifar10("train"))
DatasetCatalog.register("cifar10_test", _make_cifar10("test"))
MetadataCatalog.register("cifar10_train", dict(_CIFAR10_METADATA))
MetadataCatalog.register("cifar10_test", dict(_CIFAR10_METADATA))