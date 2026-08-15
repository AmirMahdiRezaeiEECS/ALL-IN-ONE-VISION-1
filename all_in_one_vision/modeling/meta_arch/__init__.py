"""
Every class exported here is a complete, trainable model following the
shared forward contract described in mlp.py's module docstring: a dict
of named losses in training mode, raw logits in eval mode. That shared
contract is what lets engine/ and evaluation/ work with any of them
unchanged. Adding a new one = write the class + add an import line here.
See docs/04_extending_the_project.md ("Adding a new model").
"""
from .mlp import SimpleMLPClassifier
from .cnn import SimpleCNNClassifier

__all__ = ["SimpleMLPClassifier", "SimpleCNNClassifier"]
