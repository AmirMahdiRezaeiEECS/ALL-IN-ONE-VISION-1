"""
DatasetCatalog / MetadataCatalog
=================================

WHY THIS EXISTS
----------------
Without a catalog, a config would have to know exactly how to construct a
dataset object (which class, which root path, which download flag, ...).
That couples every config to dataset internals and makes it hard to add a
new dataset without touching training code.

Instead, a dataset is registered ONCE, under a plain string name:

    DatasetCatalog.register("mnist_train", lambda transform: MNIST(...))

...and everywhere else (configs, build.py) only ever refers to it by that
name: "mnist_train". Adding CIFAR-10 later means writing
`datasets/cifar10.py` and registering "cifar10_train" / "cifar10_test" --
nothing in engine/, evaluation/, or existing configs needs to change.

MetadataCatalog stores small pieces of *information about* a dataset
(currently just num_classes / class_names for classification) that
evaluators and models may need but that isn't part of the dataset object
itself.

This mirrors Detectron2's DatasetCatalog / MetadataCatalog pattern
directly, adapted for classification (no image/annotation dict format
needed here -- a "dataset" is just anything that supports the standard
torch.utils.data.Dataset interface).
"""


class _Catalog:
    """A simple name -> factory registry, shared logic for both catalogs."""

    def __init__(self, kind: str):
        self._kind = kind
        self._registry = {}

    def register(self, name: str, factory):
        if name in self._registry:
            raise KeyError(f"{self._kind} '{name}' is already registered!")
        self._registry[name] = factory

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(
                f"{self._kind} '{name}' is not registered. "
                f"Available: {sorted(self._registry.keys())}"
            )
        return self._registry[name]

    def list(self):
        return sorted(self._registry.keys())


class _DatasetCatalog(_Catalog):
    """
    Stores dataset FACTORY functions, not dataset objects.

    Each registered factory has the signature `factory(transform) ->
    torch.utils.data.Dataset`, so the actual transform (which comes from
    the config, and can differ between experiments) is only applied when
    the dataset is actually built -- see data/build.py.
    """

    def __init__(self):
        super().__init__(kind="Dataset")


class _MetadataCatalog(_Catalog):
    """Stores small metadata dicts, e.g. {"num_classes": 10, "class_names": [...]}."""

    def __init__(self):
        super().__init__(kind="Metadata")

    def register(self, name: str, metadata: dict):
        super().register(name, metadata)


DatasetCatalog = _DatasetCatalog()
MetadataCatalog = _MetadataCatalog()
