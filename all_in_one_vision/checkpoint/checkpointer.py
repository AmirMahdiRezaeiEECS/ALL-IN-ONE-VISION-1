"""
Checkpointer
=============
REUSE-FIRST: fvcore.common.checkpoint.Checkpointer already implements
robust save/load (including optimizer state, resuming from the latest
checkpoint, etc.) -- exactly the class Detectron2 itself uses. We don't
reimplement any of that.

This subclass exists as a named, importable hook for classification-
specific behavior later (e.g. translating an old checkpoint's state_dict
keys after a model refactor). For v1, it adds nothing.
"""
from fvcore.common.checkpoint import Checkpointer as _Checkpointer


class Checkpointer(_Checkpointer):
    pass
