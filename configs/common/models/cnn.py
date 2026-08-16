"""
Model config: SimpleCNNClassifier
====================================
Same role as configs/common/models/mlp.py, but describing the CNN
variant instead. See that file's docstring for what a "building block"
config is and how LazyCall works.

GENERALIZED FOR MULTIPLE DATASETS: this now exposes `in_channels` and
`image_size` (in addition to `num_classes`) as overridable fields, so
the same building block serves both MNIST (1x28x28, the defaults below)
and CIFAR-10 (3x32x32, overridden in configs/CIFAR-10/cnn_baseline.py)
without needing a separate cnn_cifar10.py file. This mirrors how
model.hidden_dim is already overridable for the MLP -- see
docs/04_extending_the_project.md.

SWAPPING MODELS IS A ONE-LINE CHANGE: any experiment config that imports
`model` from mlp.py can switch to the CNN by importing it from HERE
instead -- nothing else in that experiment config needs to change.
Compare configs/MNIST/mlp_baseline.py and configs/MNIST/cnn_baseline.py:
they differ only in this one import line.
"""
from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.modeling.meta_arch import SimpleCNNClassifier

model = L(SimpleCNNClassifier)(
    in_channels=1,   # 1 = grayscale (MNIST). Override to 3 for RGB datasets (e.g. CIFAR-10).
    num_classes=10,  # 10 classes
    image_size=28,   # side length of the square input image (MNIST=28, CIFAR-10=32)
)