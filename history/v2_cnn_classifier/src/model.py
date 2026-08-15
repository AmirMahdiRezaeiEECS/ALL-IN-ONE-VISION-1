"""
model.py
========

WHAT THIS FILE DOES (in plain English)
---------------------------------------
This is v2's "brain" — a small Convolutional Neural Network (CNN) that
replaces v1's MLP. This is the ONLY file that changed from v1: dataset.py,
train.py, and evaluate.py are untouched, because they only care that the
model behaves like a standard PyTorch model (image in, 10 logits out).
That boundary was designed into v1 on purpose, and it holds here.

------------------------------------------------------------------
BACKGROUND CONCEPT #1: Why a CNN instead of an MLP?
------------------------------------------------------------------
v1's MLP flattened each 28x28 image into a flat list of 784 numbers before
looking at it — throwing away all 2D spatial structure (which pixels are
NEXT TO which other pixels). A CNN instead slides small learnable filters
("kernels") across the image, so it can directly learn spatial patterns
like edges, curves, and strokes — the actual building blocks of a
handwritten digit. This is why CNNs are the standard architecture for
image data, and why this is the natural "smallest effective" upgrade
from v1.

------------------------------------------------------------------
BACKGROUND CONCEPT #2: What is a Conv2d layer?
------------------------------------------------------------------
`nn.Conv2d(in_channels, out_channels, kernel_size)` slides a small
`kernel_size x kernel_size` window over the image. At each position, it
computes a weighted sum of the pixels under the window (the weights are
the learnable kernel), producing one output value per position. Doing
this for `out_channels` different kernels in parallel gives you
`out_channels` different "feature maps" — each one highlighting a
different learned pattern (e.g. one kernel might learn to detect vertical
edges, another horizontal edges, etc.). `padding=1` with a 3x3 kernel
keeps the output the same height/width as the input, so we don't lose
pixels at the borders.

------------------------------------------------------------------
BACKGROUND CONCEPT #3: What is MaxPool2d, and why use it?
------------------------------------------------------------------
`nn.MaxPool2d(2)` looks at each non-overlapping 2x2 block of the feature
map and keeps only the largest value, halving both height and width. This
does two things: (1) reduces the amount of computation needed in later
layers, and (2) gives the network a small amount of tolerance to exactly
where a feature appears (a stroke shifted by 1 pixel still triggers
roughly the same output). This is standard practice, not unique to this
project — reuse-first applies to architectural building blocks too.

------------------------------------------------------------------
BACKGROUND CONCEPT #4: The overall shape of this network
------------------------------------------------------------------
    input image  (1, 28, 28)
        -> Conv2d(1  -> 16 channels, 3x3, padding=1)  -> (16, 28, 28)
        -> ReLU
        -> MaxPool2d(2)                                -> (16, 14, 14)
        -> Conv2d(16 -> 32 channels, 3x3, padding=1)  -> (32, 14, 14)
        -> ReLU
        -> MaxPool2d(2)                                -> (32, 7, 7)
        -> Flatten                                     -> (32*7*7 = 1568)
        -> Linear(1568 -> 10)                          -> (10 logits)

This is intentionally small and simple — two conv blocks are already
more than enough to comfortably beat the v1 MLP on MNIST. Deeper/wider
CNNs, batch norm, dropout, data augmentation, etc. are all explicitly
deferred; they'd be over-engineering for what is still a "prove the
architecture-swap boundary works" version.
"""

import torch.nn as nn


class CNNClassifier(nn.Module):
    """
    A small CNN: two Conv+ReLU+MaxPool blocks, followed by one Linear
    layer mapping the final flattened feature map to 10 class logits.

    Kept as the drop-in replacement for v1's MLPClassifier: same
    constructor-free-of-surprises usage (`model = CNNClassifier()`),
    same call signature (`model(images)` -> logits of shape
    (batch_size, 10)), so main.py, train.py, and evaluate.py did not
    need to change at all.
    """

    def __init__(self, num_classes: int = 10):
        """
        Parameters
        ----------
        num_classes : int
            Number of output categories. Fixed at 10 for MNIST digits
            (0 through 9), matching v1.
        """
        super().__init__()

        # First conv block: 1 input channel (grayscale) -> 16 feature maps.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Second conv block: 16 -> 32 feature maps. Deeper layers typically
        # use MORE channels, since they're combining simpler features from
        # the previous layer into more complex, higher-level ones.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)  # halves height and width each time

        self.flatten = nn.Flatten()

        # After conv1+pool: 28x28 -> 14x14. After conv2+pool: 14x14 -> 7x7.
        # Final feature map is (32 channels, 7, 7) -> 32*7*7 = 1568 values
        # per image once flattened.
        self.classifier = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            A batch of images, shape (batch_size, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            A batch of raw logits, shape (batch_size, 10) — same
            contract as v1's MLPClassifier, which is exactly why
            train.py and evaluate.py needed zero changes.
        """
        x = self.pool(self.relu(self.conv1(x)))  # (batch, 16, 14, 14)
        x = self.pool(self.relu(self.conv2(x)))  # (batch, 32, 7, 7)
        x = self.flatten(x)                       # (batch, 1568)
        return self.classifier(x)                 # (batch, 10)
