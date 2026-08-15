"""
Experiment: MLP baseline on MNIST
=====================================
This is a COMPLETE experiment description -- it composes four building
blocks (model, data, optimizer, train settings) from configs/common/,
then tweaks a few fields for this specific run. This is the file you
point `tools/train_net.py` at:

    python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py

This is also the config that reproduces the original v1 standalone
script's setup (a single-hidden-layer MLP, SGD, 5 epochs), just expressed
declaratively instead of as a hardcoded script.

WANT TO TRY THE CNN INSTEAD? See configs/MNIST/cnn_baseline.py -- it's
identical except for one import line. WANT DIFFERENT HYPERPARAMETERS
WITHOUT A NEW FILE? Override from the command line instead, e.g.:

    python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py \
        train.max_epochs=10 optimizer.lr=0.005

See docs/00_start_here.md for a guided first run, and
docs/03_training_workflow_walkthrough.md for exactly what happens when
this file is loaded and run.
"""
from configs.common.models.mlp import model
from configs.common.data.mnist import dataloader
from configs.common.optim import optimizer
from configs.common.train import train

# Overrides specific to THIS experiment (everything else is inherited
# as-is from the common/ building blocks imported above).
train.output_dir = "./output/mnist_mlp_v1"
train.max_epochs = 5
train.mlflow.experiment_name = "all-in-one-vision-mnist"
