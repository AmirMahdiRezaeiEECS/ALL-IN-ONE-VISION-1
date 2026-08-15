"""
SimpleMLPClassifier
=====================
This is your v1 model, ported into the new architecture. The math is
unchanged (flatten -> linear -> ReLU -> linear -> logits); what's new is
the forward() CONTRACT, which follows Detectron2's meta-arch convention:

  - In TRAINING mode (self.training == True): forward() is given the
    targets too, and returns a dict of named losses:
        {"loss_cls": <scalar tensor>}
    The trainer sums this dict and calls .backward() on the sum -- see
    engine/train_loop.py. Returning a *dict* (not just a scalar) is what
    lets a future, more complex model return multiple loss terms (e.g.
    a classification loss + an auxiliary loss) without changing the
    trainer at all.

  - In EVAL mode (self.training == False): forward() is given only the
    images and returns raw logits, shape (batch, num_classes). This is
    what evaluation/classification_evaluation.py consumes.

Why keep this contract instead of just always returning logits? Because
it's the same contract every Detectron2 meta-arch (GeneralizedRCNN,
RetinaNet, ...) follows, and keeping it means engine/, evaluation/, and
tools/train_net.py don't need any classification-specific special-casing
-- they'd work unchanged even for a very different classification model.
"""
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLPClassifier(nn.Module):
    """
    A minimal fully-connected classifier: Flatten -> Linear -> ReLU -> Linear.

    Args:
        in_features: number of input pixels when flattened (28*28=784 for MNIST).
        hidden_dim: width of the single hidden layer.
        num_classes: number of output classes (10 for MNIST digits).
    """

    def __init__(self, in_features: int = 28 * 28, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, images, targets=None):
        x = self.flatten(images)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        if self.training:
            assert targets is not None, "targets are required in training mode"
            loss = F.cross_entropy(logits, targets)
            return {"loss_cls": loss}

        return logits
