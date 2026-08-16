"""
Importing this package registers all built-in datasets as a SIDE EFFECT
(each submodule calls DatasetCatalog.register(...) at import time -- see
mnist.py). Adding a new dataset = add a new module here + one import
line below. Nothing that refers to datasets by name elsewhere needs to
change. See docs/04_extending_the_project.md ("Adding a new dataset").
"""
from . import mnist    # noqa: F401
from . import cifar10  # noqa: F401

__all__ = ["mnist", "cifar10"]