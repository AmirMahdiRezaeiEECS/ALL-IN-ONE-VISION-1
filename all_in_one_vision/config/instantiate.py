"""
instantiate()
==============

The counterpart to LazyCall. Walks a config tree and, wherever it finds a
dict containing "_target_", resolves that string back into a real
class/function and calls it with the (recursively instantiated) remaining
keys as keyword arguments.

Plain values (ints, strings, lists without a _target_, already-built
objects) are returned unchanged. This is what lets a fully-resolved config
tree be turned into real model/optimizer/dataloader objects in one call:

    cfg = LazyConfig.load("configs/MNIST/mlp_baseline.py")
    model = instantiate(cfg.model)          # -> a real nn.Module
    loader = instantiate(cfg.dataloader.train)  # -> a real DataLoader
"""
import importlib

from omegaconf import DictConfig, ListConfig


def _locate(import_path: str):
    """
    Resolve a dotted import path string (e.g.
    "all_in_one_vision.modeling.meta_arch.mlp.SimpleMLPClassifier") back
    into the actual Python object it refers to.
    """
    module_name, _, attr_name = import_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Cannot locate object from path: {import_path!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise ImportError(f"Cannot find {attr_name!r} in module {module_name!r}") from e


def instantiate(cfg):
    """
    Recursively resolve a config node into real objects.

    - dict/DictConfig WITH "_target_"  -> resolve target, instantiate kwargs
                                            recursively, call target(**kwargs)
    - dict/DictConfig WITHOUT "_target_" -> plain dict, recurse into values
    - list/ListConfig                    -> recurse into elements
    - anything else (int, str, tensor, already-built object, ...) -> as-is
    """
    if isinstance(cfg, (list, ListConfig)):
        return [instantiate(x) for x in cfg]

    if isinstance(cfg, (dict, DictConfig)):
        if "_target_" in cfg:
            target = cfg["_target_"]
            cls_or_fn = _locate(target) if isinstance(target, str) else target
            kwargs = {k: instantiate(v) for k, v in cfg.items() if k != "_target_"}
            try:
                return cls_or_fn(**kwargs)
            except TypeError as e:
                raise TypeError(
                    f"Error instantiating {target} with kwargs {list(kwargs.keys())}: {e}"
                ) from e
        else:
            return {k: instantiate(v) for k, v in cfg.items()}

    return cfg
