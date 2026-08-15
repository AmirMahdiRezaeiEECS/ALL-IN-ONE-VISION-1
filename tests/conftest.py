"""
Registers a tiny synthetic dataset ("fake_mnist_train"/"fake_mnist_test")
in the DatasetCatalog for tests, so the test suite never needs real
network access / a real MNIST download.
"""
import torch
from torch.utils.data import Dataset

from all_in_one_vision.data.catalog import DatasetCatalog, MetadataCatalog


class _FakeMNIST(Dataset):
    def __init__(self, n=32, transform=None):
        self.n = n
        self.transform = transform
        self.images = torch.rand(n, 1, 28, 28)
        self.labels = torch.randint(0, 10, (n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            pass  # already a tensor; skip PIL-based transform for the fake data
        return img, self.labels[idx].item()


def _register_fake_mnist():
    if "fake_mnist_train" in DatasetCatalog.list():
        return
    DatasetCatalog.register("fake_mnist_train", lambda transform: _FakeMNIST(64, transform))
    DatasetCatalog.register("fake_mnist_test", lambda transform: _FakeMNIST(32, transform))
    MetadataCatalog.register("fake_mnist_train", {"num_classes": 10})
    MetadataCatalog.register("fake_mnist_test", {"num_classes": 10})


_register_fake_mnist()
