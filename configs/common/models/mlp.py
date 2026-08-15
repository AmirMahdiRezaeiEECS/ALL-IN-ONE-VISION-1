"""
Model config: SimpleMLPClassifier
====================================
This file is a "building block" config -- it describes ONE piece (the
model) so that experiment configs (e.g. configs/MNIST/mlp_baseline.py)
can import and combine it with other building blocks (data, optimizer,
training settings) without repeating this description every time.

WHAT `model` ACTUALLY IS: not a real model object. `L(SimpleMLPClassifier)(...)`
(LazyCall) records "build a SimpleMLPClassifier with these arguments" as
plain data -- the real nn.Module only gets constructed later, when
`instantiate(cfg.model)` is called (see tools/train_net.py). Full
explanation: docs/02_config_system_deep_dive.md.

HOW TO CHANGE THE MODEL'S SIZE: edit the numbers below directly, or
override from the command line without touching this file at all:
    python tools/train_net.py --config-file configs/MNIST/mlp_baseline.py \
        model.hidden_dim=256
"""
from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.modeling.meta_arch import SimpleMLPClassifier

model = L(SimpleMLPClassifier)(
    in_features=28 * 28,  # 28x28 pixels flattened into one vector (MNIST image size)
    hidden_dim=128,        # width of the single hidden layer
    num_classes=10,        # 10 digits (0-9)
)
