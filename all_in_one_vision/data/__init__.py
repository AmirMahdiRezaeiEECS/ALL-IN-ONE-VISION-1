"""
data
======
Dataset registration and loading. See catalog.py for the DatasetCatalog
pattern, build.py for how a catalog entry becomes a real DataLoader, and
datasets/ for the built-in dataset registrations (currently: MNIST).
Deep dive: docs/01_architecture_and_concepts.md ("The Catalog pattern").
"""
from .catalog import DatasetCatalog, MetadataCatalog
from .build import build_loader
from . import datasets  # noqa: F401  (registers built-in datasets as a side effect)

__all__ = ["DatasetCatalog", "MetadataCatalog", "build_loader"]
