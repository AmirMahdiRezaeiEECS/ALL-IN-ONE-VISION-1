"""
solver
========
Optimizer-construction helpers. See build.py's docstring for why
building an optimizer from a declarative config needs a small extra step
that most other components don't (the model has to exist first).
"""
from .build import get_default_optimizer_params

__all__ = ["get_default_optimizer_params"]
