"""
train.py
========

WHAT THIS FILE DOES (in plain English)
---------------------------------------
This is where actual "learning" happens. It repeatedly:
    1. Shows the model a batch of images.
    2. Checks how wrong its guesses were (the "loss").
    3. Automatically nudges the model's internal numbers (weights/biases)
       to make it slightly less wrong next time.
    4. Repeats this for every batch, for several full passes over the
       data (each full pass = one "epoch").

------------------------------------------------------------------
BACKGROUND CONCEPT #1: What is "loss"?
------------------------------------------------------------------
"Loss" is a single number that measures how wrong the model's predictions
were for a batch — lower is better, 0 would mean perfect predictions.
We use `CrossEntropyLoss` (defined in main.py, passed into this file),
which is the standard loss function for classification problems like
this one (choosing 1 correct answer out of several categories). You don't
need to know its exact formula — just know: it punishes confident WRONG
answers more harshly than unsure wrong answers, which is exactly the
behavior we want while training a classifier.

------------------------------------------------------------------
BACKGROUND CONCEPT #2: What is backpropagation / `loss.backward()`?
------------------------------------------------------------------
This is the single most important trick that makes deep learning work.

After computing the loss for a batch, we need to know: "if I nudge THIS
particular weight up or down a little, does the loss get better or
worse, and by how much?" That sensitivity value is called a "gradient".

`loss.backward()` automatically calculates the gradient for EVERY single
weight and bias in the entire model in one call, using calculus (the
chain rule) applied backwards through the network — hence the name
"backpropagation". You never need to compute these derivatives by hand;
PyTorch's "autograd" system handles it entirely automatically.

------------------------------------------------------------------
BACKGROUND CONCEPT #3: What is an "optimizer" and `optimizer.step()`?
------------------------------------------------------------------
Once we know the gradients (from backward()), something needs to actually
USE them to update the weights. That's the optimizer's job.

The simplest possible update rule would be:
    new_weight = old_weight - (learning_rate × gradient)

This nudges every weight slightly in the direction that reduces loss.
The "learning_rate" controls how big each nudge is — too large and
training becomes unstable/erratic; too small and training takes forever.

We use `Adam` (defined in main.py), a more advanced optimizer that
automatically adapts the step size per-weight based on recent gradient
history. It's a well-tested, reliable default choice for most projects
— reuse-first applies here too: no need to invent a custom optimizer.

------------------------------------------------------------------
BACKGROUND CONCEPT #4: Why `optimizer.zero_grad()`?
------------------------------------------------------------------
By default, PyTorch ACCUMULATES (adds up) gradients every time you call
`.backward()`, rather than replacing them. This is useful in some
advanced scenarios, but for standard training we want a FRESH gradient
calculation for each new batch. So before computing gradients for a new
batch, we explicitly clear out ("zero") whatever gradients are left over
from the previous batch. Forgetting this line is one of the most common
beginner bugs in PyTorch code!

------------------------------------------------------------------
BACKGROUND CONCEPT #5: What is an "epoch"?
------------------------------------------------------------------
One epoch = one complete pass through the ENTIRE training dataset (all
60,000 images, split into batches). We do MULTIPLE epochs (5, in this
project) because a single pass usually isn't enough for the model to
learn well — repeated exposure (with reshuffled batch order each time)
lets it gradually refine its weights further.

------------------------------------------------------------------
PUTTING IT ALL TOGETHER: the training loop, step by step
------------------------------------------------------------------
For each epoch:
    For each batch of (images, labels) in the training data:
        1. optimizer.zero_grad()      -> clear old gradients
        2. outputs = model(images)    -> forward pass: get predictions
        3. loss = criterion(outputs, labels)  -> how wrong were we?
        4. loss.backward()            -> compute new gradients
        5. optimizer.step()           -> nudge the weights using those gradients
    (then print progress and move to the next epoch)
"""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs ONE full pass over the training data (one epoch) and returns
    the average loss and accuracy achieved during that pass.

    Parameters
    ----------
    model : nn.Module
        The neural network being trained (e.g. our MLPClassifier).
    loader : DataLoader
        Yields batches of (images, labels) to train on.
    optimizer : torch.optim.Optimizer
        Updates the model's weights using computed gradients (e.g. Adam).
    criterion : loss function
        Measures how wrong the model's predictions are (e.g. CrossEntropyLoss).
    device : torch.device
        Whether we're computing on "cpu" or "cuda" (GPU). Both the model
        and the data need to be on the SAME device, or PyTorch will error.

    Returns
    -------
    avg_loss : float
        Average loss across all batches this epoch (lower = better).
    accuracy : float
        Fraction of images correctly classified this epoch (0.0 to 1.0).
    """

    # model.train() puts the model in "training mode". For this simple
    # MLP it doesn't change behavior, but some layers (like Dropout or
    # BatchNorm, used in more advanced models) behave differently during
    # training vs. evaluation — calling this explicitly is good habit-
    # forming practice for when the model grows more complex in v2+.
    model.train()

    running_loss = 0.0  # accumulates total loss across all images this epoch
    correct = 0          # counts how many predictions were correct
    total = 0             # counts how many images we've processed so far

    # The DataLoader automatically hands us one batch at a time. Each
    # batch is a pair: a stack of images, and their matching true labels.
    for images, labels in loader:
        # Move this batch's data onto whichever device (CPU/GPU) the
        # model itself lives on, so PyTorch can do the math on them together.
        images, labels = images.to(device), labels.to(device)

        # Step 1: clear gradients left over from the previous batch.
        optimizer.zero_grad()

        # Step 2: "forward pass" — ask the model for its predictions
        # (raw logits) on this batch of images.
        outputs = model(images)

        # Step 3: compare predictions to the true labels and compute
        # a single "how wrong were we" loss number for this batch.
        loss = criterion(outputs, labels)

        # Step 4: backpropagation — automatically compute how every
        # single weight in the model contributed to this loss.
        loss.backward()

        # Step 5: use those gradients to actually update the model's
        # weights, nudging them to reduce the loss next time.
        optimizer.step()

        # --- Everything below is just bookkeeping for progress reporting,
        #     it does NOT affect training itself. ---

        # loss.item() converts the loss from a PyTorch tensor into a
        # plain Python number. We multiply by the batch size because
        # `loss` is already an AVERAGE over the batch, and we want a
        # running TOTAL across all images to compute an overall average
        # at the end (batches can vary slightly in size, especially the
        # last one, so this keeps the average precisely correct).
        running_loss += loss.item() * images.size(0)

        # For each image, the model's guess is whichever of the 10
        # output logits is highest ("argmax" = index of the max value).
        predictions = outputs.argmax(dim=1)

        # Count how many of those guesses matched the true label.
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train(model, train_loader, optimizer, criterion, device, epochs: int):
    """
    Runs the FULL training process: `epochs` complete passes over the
    training data, printing progress after each one so you can watch the
    loss go down and accuracy go up over time.

    Parameters
    ----------
    epochs : int
        How many times to loop over the entire training dataset.
    (see train_one_epoch() above for the other parameters)
    """
    for epoch in range(1, epochs + 1):
        avg_loss, accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{epochs} | train_loss={avg_loss:.4f} | train_acc={accuracy:.4f}")
        # As training progresses over successive epochs, you should see
        # train_loss trend DOWN and train_acc trend UP. If it plateaus or
        # gets worse, that's usually a sign to investigate (e.g. learning
        # rate too high, or too many epochs causing overfitting) — but for
        # this simple MNIST MLP baseline, steady improvement is expected.
