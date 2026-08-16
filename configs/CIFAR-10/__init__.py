"""
Experiment configs for CIFAR-10. Each file here is a complete, runnable
experiment -- point tools/train_net.py's --config-file at one:

    python tools/train_net.py --config-file configs/CIFAR-10/cnn_baseline.py

No MLP baseline for CIFAR-10 (deliberate -- CNN v1 only for this
dataset; a flat MLP on 3x32x32 RGB images isn't a useful baseline here
the way it was for MNIST).
"""