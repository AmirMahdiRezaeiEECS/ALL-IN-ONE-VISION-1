"""
Model config: SimpleCNNClassifier
====================================
Same role as configs/common/models/mlp.py, but describing the CNN
variant instead. See that file's docstring for what a "building block"
config is and how LazyCall works.

SWAPPING MODELS IS A ONE-LINE CHANGE: any experiment config that imports
`model` from mlp.py can switch to the CNN by importing it from HERE
instead -- nothing else in that experiment config needs to change.
Compare configs/MNIST/mlp_baseline.py and configs/MNIST/cnn_baseline.py:
they differ only in this one import line. See
docs/04_extending_the_project.md for the full "adding a new model" guide.
"""
from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.modeling.meta_arch import SimpleCNNClassifier

model = L(SimpleCNNClassifier)(
    in_channels=1,   # 1 = grayscale (MNIST). Use 3 for RGB datasets.
    num_classes=10,  # 10 digits (0-9)
)
