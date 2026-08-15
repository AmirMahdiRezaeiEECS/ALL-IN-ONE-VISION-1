"""
LazyConfig
==========

Loads a *Python* config file (e.g. configs/MNIST/mlp_baseline.py) and
collects its top-level variables (model, dataloader, optimizer, train, ...)
into a single OmegaConf tree that the rest of the codebase can consume.

DESIGN NOTE / DEVIATION FROM UPSTREAM DETECTRON2 (flagged explicitly,
see chat discussion): Detectron2's real LazyConfig patches Python's import
machinery so that config files can `from .common.optim import optimizer`
using *relative* imports even though they aren't a real installed package.
That machinery is one of the more intricate parts of Detectron2 and adds
real complexity for a benefit we don't need yet.

Here, `configs/` is instead a plain, ordinary Python package (every folder
has an `__init__.py`) sitting at the project root. Config files use normal
ABSOLUTE imports:

    from configs.common.models.mlp import model
    from configs.common.data.mnist import dataloader

This keeps every other part of the LazyConfig philosophy identical
(LazyCall recipes, instantiate(), dotted-key CLI overrides) while relying
on Python's ordinary, well-understood import system instead of custom
import-patching. If we ever need config files to live outside the
installed package (e.g. user-supplied configs outside the repo), we can
revisit this and add the upstream-style import patching then.
"""
import ast
import types
from pathlib import Path

import yaml
from omegaconf import OmegaConf


class LazyConfig:
    @staticmethod
    def load(filename: str):
        """
        Execute a config .py file and collect its top-level, "public"
        variables (anything not starting with "_" and not itself an
        imported module) into an OmegaConf tree.
        """
        filename = str(Path(filename).resolve())
        with open(filename) as f:
            src = f.read()

        # Fail fast with a clear error on malformed config files, before
        # even attempting to exec them.
        ast.parse(src, filename=filename)

        module_namespace = {"__file__": filename, "__name__": "all_in_one_vision_config"}
        exec(compile(src, filename, "exec"), module_namespace)

        cfg_dict = {
            k: v
            for k, v in module_namespace.items()
            if not k.startswith("_") and not isinstance(v, types.ModuleType)
        }
        return OmegaConf.create(cfg_dict)

    @staticmethod
    def apply_overrides(cfg, overrides):
        """
        Apply CLI-style dotted-key overrides, e.g.
            ["train.max_epochs=10", "optimizer.lr=0.001"]
        onto a loaded config, in place, and return it.

        Values are parsed with YAML so "10" -> int, "0.001" -> float,
        "true" -> bool, etc., without the caller needing to worry about types.
        """
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override (expected key=value): {override!r}")
            key, value_str = override.split("=", 1)
            value = yaml.safe_load(value_str)
            OmegaConf.update(cfg, key.strip(), value, merge=True)
        return cfg
