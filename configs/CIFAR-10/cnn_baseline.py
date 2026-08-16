"""
Experiment: CNN baseline on CIFAR-10
=====================================
Same shape as configs/MNIST/cnn_baseline.py -- compare the two side by
side. The differences: the data building block (cifar10 instead of
mnist) and two model-field overrides (in_channels=3, image_size=32) to
match CIFAR-10's 3x32x32 RGB images, since configs/common/models/cnn.py
now exposes both as overridable fields instead of being MNIST-only.

Deliberately kept as simple as the original CNN v1 (no BatchNorm,
dropout, or extra layers) -- see cnn.py's docstring. This is a v1
baseline; improvements come later, driven by whatever the actual
bottleneck turns out to be once this runs.

    python tools/train_net.py --config-file configs/CIFAR-10/cnn_baseline.py
"""
from configs.common.models.cnn import model
from configs.common.data.cifar10 import dataloader
from configs.common.optim import optimizer
from configs.common.train import train

model.in_channels = 3
model.image_size = 32

train.output_dir = "./output/cifar10_cnn_v1"
train.max_epochs = 5
train.mlflow.experiment_name = "all-in-one-vision-cifar10"