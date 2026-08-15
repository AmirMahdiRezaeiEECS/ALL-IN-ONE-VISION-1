"""
Experiment: CNN baseline on MNIST
=====================================
Identical in structure to configs/MNIST/mlp_baseline.py -- compare the
two files side by side. The ONLY difference is the first import line
(`configs.common.models.cnn` instead of `configs.common.models.mlp`).
Everything else -- data, optimizer, training settings, even how to run
it -- is exactly the same, which is the whole point: swapping models is
a config-only change. See docs/04_extending_the_project.md.

    python tools/train_net.py --config-file configs/MNIST/cnn_baseline.py
"""
from configs.common.models.cnn import model
from configs.common.data.mnist import dataloader
from configs.common.optim import optimizer
from configs.common.train import train

train.output_dir = "./output/mnist_cnn_v1"
train.max_epochs = 5
train.mlflow.experiment_name = "all-in-one-vision-mnist"
