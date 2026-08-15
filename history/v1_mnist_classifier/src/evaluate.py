"""
evaluate.py
===========

WHAT THIS FILE DOES (in plain English)
---------------------------------------
Training makes the model change its behavior; evaluation just MEASURES
how good that behavior is, without changing anything. Crucially, we
measure on the TEST set — images the model has never seen during
training — because that's the only fair way to estimate how well it
would perform on brand new, real-world digits.

------------------------------------------------------------------
BACKGROUND CONCEPT #1: Why a separate test set at all?
------------------------------------------------------------------
If we only measured accuracy on the same images the model trained on, a
model could theoretically just "memorize" those exact images (including
their noise/quirks) rather than learning the general PATTERN of what
makes a "3" look like a "3". This is called "overfitting". Measuring on
held-out test images (that had zero influence on training) is how we
catch that problem and get an honest read on real-world performance.

------------------------------------------------------------------
BACKGROUND CONCEPT #2: What does `@torch.no_grad()` do, and why?
------------------------------------------------------------------
During training, PyTorch tracks every mathematical operation on tensors
so it can later compute gradients (see train.py's backpropagation
explanation). That tracking costs extra memory and computation.

During evaluation, we're NOT going to call `.backward()` or update any
weights — we're just checking predictions. So we tell PyTorch "don't
bother tracking gradients for anything in this function" using the
`@torch.no_grad()` decorator. This makes evaluation notably faster and
uses less memory, with zero downside, since we don't need gradients here.

------------------------------------------------------------------
BACKGROUND CONCEPT #3: What is a confusion matrix?
------------------------------------------------------------------
A single "accuracy" number (e.g. "97% correct") is useful, but it hides
WHERE the model is making mistakes. Maybe it's great at every digit
except constantly confusing 4s with 9s (which do look visually similar!).

A confusion matrix is a grid that shows exactly this:
    - Each ROW represents the TRUE digit.
    - Each COLUMN represents what the model PREDICTED.
    - Each cell counts how many test images fall into that (true,
      predicted) combination.

A perfect model would have all its counts along the diagonal (where
true label == predicted label) and zeros everywhere else. Numbers off
the diagonal reveal specific, systematic mistakes — e.g. a large number
in row "4", column "9" means the model often mistakes 4s for 9s.
"""

import torch


@torch.no_grad()  # disable gradient tracking for everything in this function (see above)
def evaluate(model, loader, device, num_classes: int = 10):
    """
    Runs the model over an entire dataset (typically the test set) WITHOUT
    training it, and reports how accurate it is, plus a full breakdown of
    which digits get confused with which.

    Parameters
    ----------
    model : nn.Module
        The (already trained) model to evaluate.
    loader : DataLoader
        Yields batches of (images, labels) to evaluate on.
    device : torch.device
        "cpu" or "cuda" — must match where the model lives.
    num_classes : int
        Number of possible categories (10 for MNIST digits).

    Returns
    -------
    accuracy : float
        Fraction of test images correctly classified (0.0 to 1.0).
    confusion : torch.Tensor
        A (num_classes, num_classes) grid of counts; confusion[i][j] =
        number of images whose TRUE label was i but were PREDICTED as j.
    """

    # model.eval() puts the model in "evaluation mode" — the counterpart
    # to model.train() used during training. For our simple MLP this
    # doesn't change behavior, but it's the correct habit for any model
    # (this project's future CNN versions may include layers, like
    # Dropout, that DO behave differently here).
    model.eval()

    correct = 0
    total = 0

    # Start with an all-zeros grid to tally up results into.
    # dtype=torch.int64 just means "store these as whole numbers (counts)",
    # since we're counting occurrences, not measuring continuous values.
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass only — get the model's raw predictions (logits).
        outputs = model(images)

        # The model's actual "guess" for each image is whichever of the
        # 10 output logits is highest.
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # Update the confusion matrix one image at a time: for every
        # (true_label, predicted_label) pair in this batch, increment
        # the matching cell in the grid by 1.
        for true_label, predicted_label in zip(labels.view(-1), predictions.view(-1)):
            confusion[true_label.long(), predicted_label.long()] += 1

    accuracy = correct / total
    return accuracy, confusion


def print_confusion_matrix(confusion: torch.Tensor):
    """
    Pretty-prints the confusion matrix to the terminal so it's actually
    readable as a grid of numbers with row/column headers.

    How to read the output: find the row for a digit you care about
    (say, true label "4"). Reading across that row shows how the model's
    predictions were distributed for all real 4s in the test set — a
    large number in the "4" column is good (correctly predicted 4), and
    any large numbers in OTHER columns reveal a specific, systematic
    mistake (e.g. mistaking many 4s for 9s).
    """
    print("\nConfusion matrix (rows = true label, cols = predicted label):")
    header = "     " + " ".join(f"{i:4d}" for i in range(confusion.size(0)))
    print(header)
    for i, row in enumerate(confusion):
        row_str = " ".join(f"{v.item():4d}" for v in row)
        print(f"{i:4d} {row_str}")
