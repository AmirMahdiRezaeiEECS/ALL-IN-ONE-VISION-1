import pytest

from all_in_one_vision.data import DatasetCatalog, build_loader


def test_mnist_is_registered_on_import():
    names = DatasetCatalog.list()
    assert "mnist_train" in names
    assert "mnist_test" in names


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        DatasetCatalog.get("does_not_exist")


def test_build_loader_with_fake_dataset():
    loader = build_loader(
        dataset_name="fake_mnist_train", transform=None, batch_size=8, shuffle=False
    )
    images, labels = next(iter(loader))
    assert images.shape == (8, 1, 28, 28)
    assert labels.shape == (8,)

def test_cifar10_is_registered_on_import():
    names = DatasetCatalog.list()
    assert "cifar10_train" in names
    assert "cifar10_test" in names