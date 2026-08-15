"""
checkpoint
============
Model/optimizer checkpoint saving and loading. Checkpointer here is a
thin, reuse-first subclass of fvcore's own Checkpointer -- the same
class Detectron2 itself uses -- rather than a reimplementation.
"""
from .checkpointer import Checkpointer

__all__ = ["Checkpointer"]
