"""
model.py
========

WHAT THIS FILE DOES (in plain English)
---------------------------------------
This file defines the actual "brain" — the neural network architecture
that will look at an image and guess which digit (0-9) it shows.

This is the ONLY file a future version (e.g. a CNN-based v2) needs to
replace. Everything else in the project (loading data, training,
evaluating) doesn't care what's INSIDE the model, only that it behaves
like a standard PyTorch model (takes an image in, returns 10 numbers out).
That's the benefit of good separation of responsibilities.

------------------------------------------------------------------
BACKGROUND CONCEPT #1: What is a neural network, really?
------------------------------------------------------------------
At its core, a neural network is just a big mathematical function made of
stacked, simple operations, with lots of adjustable numbers inside it
called "weights" and "biases". Training the network means automatically
adjusting those numbers so that the function's output gets closer and
closer to the correct answer over time.

You can think of it like a very complicated dial-tuning process: at the
start, all the dials (weights) are set randomly, so the model's guesses
are basically random. Training nudges every dial a tiny bit, over and
over, in the direction that reduces mistakes.

------------------------------------------------------------------
BACKGROUND CONCEPT #2: What is a "Linear" (fully-connected) layer?
------------------------------------------------------------------
`nn.Linear(in_features, out_features)` is the most basic building block.
Mathematically it computes:

    output = (input × weights) + bias

- `weights` is a big grid of numbers (one for every input-output pair).
- `bias` is one extra adjustable number per output.
- Both `weights` and `bias` start out as random numbers, and get updated
  automatically during training.

Concretely: if `in_features=784` and `out_features=128`, this layer takes
784 numbers in and produces 128 numbers out, where each of those 128
outputs is a different weighted combination of all 784 inputs.

Why 784? Because our image is 28x28 = 784 pixels, and this simple model
(an MLP) "flattens" the image into one long list of 784 numbers before
feeding it in — it doesn't understand 2D spatial structure the way a
Convolutional Neural Network (CNN) would. That's a deliberate simplicity
trade-off for this first version; a CNN is planned for v2.

------------------------------------------------------------------
BACKGROUND CONCEPT #3: Why do we need ReLU (non-linearity)?
------------------------------------------------------------------
If you stack multiple Linear layers directly on top of each other with
nothing in between, mathematically it's still equivalent to just ONE
big Linear layer — you gain nothing by stacking them. To let the network
learn genuinely complex, curved patterns (not just straight-line
relationships), we insert a simple non-linear function between layers.

`ReLU` (Rectified Linear Unit) is the simplest common choice:
    ReLU(x) = x if x > 0, otherwise 0

It just zeroes out negative numbers and leaves positive numbers unchanged.
That tiny bit of "bending" of the numbers, repeated across many neurons,
is what lets neural networks approximate very complex functions.

------------------------------------------------------------------
BACKGROUND CONCEPT #4: What comes out of the model, and what is a "logit"?
------------------------------------------------------------------
Our final layer has `out_features=10` — one number per possible digit
(0 through 9). These 10 raw output numbers are called "logits". They are
NOT probabilities yet (they can be any positive or negative number).

We deliberately do NOT apply a "softmax" (the function that turns logits
into probabilities that sum to 1) inside this file. That's because
PyTorch's `nn.CrossEntropyLoss` (used in main.py) expects raw logits and
applies softmax internally, in a more numerically stable way than doing
it ourselves. This is a common PyTorch convention worth remembering.
"""

import torch.nn as nn


class MLPClassifier(nn.Module):
    """
    MLP = "Multi-Layer Perceptron" — just a fancy name for a neural
    network made of a few stacked Linear layers with non-linearities
    (ReLU) in between. This is the simplest kind of neural network and a
    good learning baseline before moving to more sophisticated
    architectures (like CNNs) in later versions.

    Architecture, step by step:
        input image (1, 28, 28)
            -> flatten to a single list of 784 numbers
            -> Linear layer: 784 numbers in, 128 numbers out
            -> ReLU (adds non-linearity)
            -> Linear layer: 128 numbers in, 10 numbers out (one per digit)
            -> (softmax + loss happens outside this class, in main.py)
    """

    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 128, num_classes: int = 10):
        """
        Parameters
        ----------
        input_size : int
            Number of input numbers the model expects. Fixed at 28*28=784
            for MNIST images (1 channel, 28x28 pixels), because we flatten
            the image before feeding it in.
        hidden_size : int
            Number of "hidden" values computed between the two Linear
            layers. This is a design choice / tunable knob, not a fixed
            requirement — bigger allows the model to represent more
            complex functions, but also risks overfitting and is slower.
            128 is a reasonable, commonly-used starting point.
        num_classes : int
            Number of possible output categories. Fixed at 10 for MNIST
            digits (0 through 9).
        """
        # Every custom PyTorch model must call the parent class's
        # __init__ first — this sets up internal bookkeeping (like
        # tracking all the learnable weights/biases) that PyTorch needs.
        super().__init__()

        # nn.Flatten() reshapes each image from (batch, 1, 28, 28) into
        # (batch, 784) — squashing the 2D grid into one long row of
        # numbers per image, since our simple Linear layers only
        # understand flat lists of numbers, not 2D grids.
        self.flatten = nn.Flatten()

        # nn.Sequential just means "run these layers one after another,
        # in this exact order" — a simple container, nothing fancier.
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),   # 784 -> 128
            nn.ReLU(),                             # non-linearity
            nn.Linear(hidden_size, num_classes),   # 128 -> 10
        )

    def forward(self, x):
        """
        Defines what happens when you call `model(some_images)`.
        PyTorch calls this method automatically — you never call
        `forward()` directly yourself, you just call the model like a
        function (e.g. `outputs = model(images)`), and Python/PyTorch
        route that call here behind the scenes.

        Parameters
        ----------
        x : torch.Tensor
            A batch of images, shape (batch_size, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            A batch of raw logits, shape (batch_size, 10) — 10 scores
            per image, one per candidate digit. Higher score = model
            thinks that digit is more likely.
        """
        x = self.flatten(x)   # (batch, 1, 28, 28) -> (batch, 784)
        return self.net(x)    # (batch, 784) -> (batch, 10)
