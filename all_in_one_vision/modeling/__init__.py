"""
modeling
==========
Model definitions ("meta-archs" in Detectron2 terminology -- complete,
top-level models, as opposed to reusable sub-components like a backbone).
See meta_arch/ for the actual model classes, and backbone/ + heads/ for
placeholder packages reserved for future swappable sub-components (see
their own README.md files for why they're currently empty).
"""
from .meta_arch import SimpleMLPClassifier, SimpleCNNClassifier

__all__ = ["SimpleMLPClassifier", "SimpleCNNClassifier"]
