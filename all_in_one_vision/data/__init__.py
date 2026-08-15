from .catalog import DatasetCatalog, MetadataCatalog
from .build import build_loader
from . import datasets  # noqa: F401  (registers built-in datasets as a side effect)

__all__ = ["DatasetCatalog", "MetadataCatalog", "build_loader"]
