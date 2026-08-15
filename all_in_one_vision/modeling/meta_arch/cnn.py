"""
SimpleCNNClassifier
=====================
A small convolutional classifier: two Conv->ReLU->MaxPool blocks followed
by a fully-connected head. This is the "v2" model in the original
project's terms, but architecturally it's just another meta-arch that
happens to also live in this MVP because it shares the exact same
forward() contract as SimpleMLPClassifier (see mlp.py for why that
contract exists) -- swapping between them is a one-line config change,
nothing else in the codebase cares which one is used.

Deliberately simple for now (no BatchNorm, no dropout, fixed kernel
sizes) -- the point of this MVP is validating the architecture end-to-end,
not squeezing out accuracy. Those refinements are natural additions for a
later version once accuracy/overfitting is the actual bottleneck.
"""
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNClassifier(nn.Module):
    """
    Args:
        in_channels: number of input image channels (1 for MNIST grayscale).
        num_classes: number of output classes.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        # 28x28 -> conv/pool -> 14x14 -> conv/pool -> 7x7
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, images, targets=None):
        x = self.pool(F.relu(self.conv1(images)))   # -> (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))          # -> (B, 32, 7, 7)
        x = x.flatten(start_dim=1)                      # -> (B, 32*7*7)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        if self.training:
            assert targets is not None, "targets are required in training mode"
            loss = F.cross_entropy(logits, targets)
            return {"loss_cls": loss}

        return logits
