"""
dataset.py
==========

WHAT THIS FILE DOES (in plain English)
---------------------------------------
Before a computer can "learn" to recognize handwritten digits, it needs
examples to learn from. This file's only job is:
    1. Get those examples (the MNIST dataset) onto the computer.
    2. Convert them into a format PyTorch's numbers-crunching code understands.
    3. Hand them back in small, organized "batches" ready for training.

It does NOT define the model, and it does NOT do any training. That
separation is deliberate — see the project README for why.

------------------------------------------------------------------
BACKGROUND CONCEPT #1: What is MNIST?
------------------------------------------------------------------
MNIST is a famous dataset of 70,000 small black-and-white images of
handwritten digits (0 through 9), each 28 pixels wide and 28 pixels tall.
- 60,000 images are for TRAINING (the model learns from these).
- 10,000 images are for TESTING (we check performance on these, since the
  model has never seen them — this simulates "real world" performance).

Each image comes with a LABEL: the digit (0-9) that the image actually
shows. This makes it a "supervised learning" problem: we have both the
question (the image) and the correct answer (the label), and we want the
model to learn the relationship between them.

------------------------------------------------------------------
BACKGROUND CONCEPT #2: What is a tensor?
------------------------------------------------------------------
A "tensor" is just PyTorch's word for a multi-dimensional array of numbers
— similar to a NumPy array. A grayscale image is naturally a 2D grid of
brightness values. PyTorch represents one MNIST image as a tensor of shape
(1, 28, 28):
    - 1  = number of color channels (1 because it's grayscale, not RGB)
    - 28 = height in pixels
    - 28 = width in pixels
Every pixel is a single number. In the raw image file, that number ranges
from 0 (black) to 255 (white).

------------------------------------------------------------------
BACKGROUND CONCEPT #3: Why normalize the pixel values?
------------------------------------------------------------------
Neural networks train much better numerically when input values are small
and centered around 0, rather than large and all-positive (like 0-255).
So we do two conversions:
    1. Divide by 255 -> pixel values become 0.0 to 1.0 (this is what
       `transforms.ToTensor()` does automatically for images).
    2. Subtract the dataset's average pixel value and divide by its
       standard deviation ("standardization" / "z-score normalization")
       -> pixel values end up centered around 0, mostly between -1 and 1.
This second step uses two numbers that are already known and published
for MNIST specifically (its average brightness and spread), so we don't
need to calculate them ourselves — see MNIST_MEAN / MNIST_STD below.

You do NOT need to understand the exact math to use this file. Just know:
"normalization" = reshaping the numbers to a range that helps the model
learn faster and more reliably. This is standard practice for basically
every neural network, not just this one.

------------------------------------------------------------------
BACKGROUND CONCEPT #4: What is a DataLoader and why "batches"?
------------------------------------------------------------------
We could feed the model one image at a time, but that's slow and makes
training unstable. Instead, we group images into small groups called
"batches" (e.g. 64 images at a time) and show the model one batch at a
time. PyTorch's `DataLoader` handles this grouping (batching), and can
also shuffle the order of the images each pass, which helps the model
learn more robustly (it prevents the model from learning the ORDER of the
training data instead of the actual patterns in the images).
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# These are the average pixel value (mean) and spread (standard deviation)
# for the ENTIRE MNIST training set, pre-calculated by the community and
# used almost universally when working with this dataset. We reuse these
# known numbers rather than recomputing them ourselves (reuse-first).
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_dataloaders(data_dir: str = "./data", batch_size: int = 64):
    """
    Downloads MNIST (only the first time you run this — after that it's
    cached on disk) and returns two DataLoaders: one for training, one for
    testing.

    Parameters
    ----------
    data_dir : str
        Folder where the raw MNIST files will be stored/cached.
    batch_size : int
        How many images to group together in one batch. 64 is a common,
        safe default — small enough to fit easily in memory, large enough
        to train efficiently.

    Returns
    -------
    train_loader : DataLoader
        Yields batches of (images, labels) for training, in random order
        each time (shuffle=True) so the model doesn't memorize order.
    test_loader : DataLoader
        Yields batches of (images, labels) for evaluation. Order doesn't
        matter here since we're not learning from it, just measuring
        accuracy, so shuffle=False.
    """

    # A "transform" is a series of steps applied to every single image as
    # it's loaded, before the model ever sees it. `transforms.Compose`
    # just chains multiple steps together in order.
    transform = transforms.Compose([
        # Step 1: Convert the raw image (which looks like a picture) into
        # a PyTorch tensor of numbers, and rescale pixel values from the
        # raw 0-255 range down to 0.0-1.0.
        transforms.ToTensor(),

        # Step 2: Standardize using MNIST's known mean/std (see the big
        # comment above). This shifts values to be centered around 0.
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    # `datasets.MNIST` is a ready-made PyTorch class that knows how to
    # download and read the MNIST files for us. This is "reuse-first" in
    # action: we don't write our own file-parsing code from scratch.
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,          # True = give us the 60,000 training images
        download=True,       # download automatically if not already cached
        transform=transform,  # apply the steps defined above to every image
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,         # False = give us the 10,000 test images instead
        download=True,
        transform=transform,
    )

    # Wrap each dataset in a DataLoader, which takes care of grouping
    # images into batches (and shuffling, for training) automatically.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # reshuffle the training data every epoch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # no need to shuffle test data
    )

    return train_loader, test_loader
