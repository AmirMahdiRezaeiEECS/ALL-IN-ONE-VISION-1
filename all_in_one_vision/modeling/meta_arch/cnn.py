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

`image_size` (ADDED for CIFAR-10): the two 2x2 max-pools each halve the
spatial dimensions, so a square input of side `image_size` becomes
`image_size // 4` per side after both pools. `fc1`'s input width is
computed from this instead of being hardcoded, which is what makes this
model reusable across datasets with different image sizes (MNIST: 28x28
-> 7x7; CIFAR-10: 32x32 -> 8x8) without touching this file again.
`image_size` must be divisible by 4 for the pooled size to be exact --
both MNIST (28) and CIFAR-10 (32) satisfy this.
"""
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNClassifier(nn.Module):
    """
    Args:
        in_channels: number of input image channels (1 for grayscale, 3 for RGB).
        num_classes: number of output classes.
        image_size: side length of the (square) input image, e.g. 28 for
            MNIST, 32 for CIFAR-10. Used only to size the fc1 layer.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        assert image_size % 4 == 0, (
            f"image_size must be divisible by 4 (two 2x2 max-pools), got {image_size}"
        )
        # 28x28 -> conv/pool -> 14x14 -> conv/pool -> 7x7  (MNIST example)
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        pooled_size = image_size // 4
        flatten_dim = 32 * pooled_size * pooled_size
        self.fc1 = nn.Linear(flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, images, targets=None):
        x = self.pool(F.relu(self.conv1(images)))   # -> (B, 16, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))          # -> (B, 32, H/4, W/4)
        x = x.flatten(start_dim=1)                      # -> (B, 32*(H/4)*(W/4))
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        if self.training:
            assert targets is not None, "targets are required in training mode"
            loss = F.cross_entropy(logits, targets)
            return {"loss_cls": loss}

        return logits