"""
Importing this package registers all built-in datasets (side effect).
Adding a new dataset later = add a new module here + one import line below.
"""
from . import mnist  # noqa: F401

__all__ = ["mnist"]
