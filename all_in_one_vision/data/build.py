"""
build_loader
=============
The one function that turns "a dataset name from the catalog" + "a
transform" + "loader hyperparameters" into a real torch DataLoader.

This function itself is usually wrapped in a LazyCall in config files, e.g.

    dataloader.train = L(build_loader)(
        dataset_name="mnist_train",
        transform=L(T.Compose)(transforms=[...]),
        batch_size=64,
        shuffle=True,
    )

so nothing about *which* dataset or *what* transform is hardcoded here --
this module only knows how to assemble the pieces it's given.
"""
from torch.utils.data import DataLoader

from .catalog import DatasetCatalog


def build_loader(dataset_name, transform, batch_size, shuffle, num_workers=0):
    dataset_factory = DatasetCatalog.get(dataset_name)
    dataset = dataset_factory(transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
