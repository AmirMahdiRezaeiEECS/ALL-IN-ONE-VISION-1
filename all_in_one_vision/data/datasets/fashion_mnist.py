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
