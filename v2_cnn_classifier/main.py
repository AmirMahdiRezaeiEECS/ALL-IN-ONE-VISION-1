"""
main.py
=======

WHAT THIS FILE DOES (in plain English)
---------------------------------------
Same conductor role as v1's main.py: ties together data, model, training,
and evaluation, and holds the run's hyperparameters in one place.

The ONLY functional difference from v1: this imports `CNNClassifier`
instead of `MLPClassifier`. Everything else — the training loop, the
evaluation code, the data loading — is byte-for-byte identical to v1.
That's the point of this version: prove that swapping architectures only
requires touching src/model.py, exactly as v1's README promised.
"""

import os
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow.entities import SpanType

from src.dataset import get_dataloaders
from src.model import CNNClassifier
from src.train import train
from src.evaluate import evaluate, print_confusion_matrix

# ============================================================
# CONFIG — all the adjustable settings for this run, in one place
# ============================================================

BATCH_SIZE = 64

# CNNs typically need fewer epochs than an MLP to reach strong accuracy
# on MNIST, since convolution already encodes useful assumptions about
# image structure. 5 epochs (same as v1) is kept for a fair, direct
# comparison between the two versions.
LEARNING_RATE = 1e-3
EPOCHS = 5

DATA_DIR = "./data"
CHECKPOINT_PATH = "./saved_models/cnn_mnist.pt"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
MLFLOW_EXPERIMENT_NAME = "aio-vision-from-scratch"


def configure_mlflow():
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if "MLFLOW_EXPERIMENT_ID" not in os.environ:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.autolog()


def main():
    configure_mlflow()
    run_pipeline()


@mlflow.trace(name="v2_cnn_mnist_pipeline", span_type=SpanType.CHAIN)
def run_pipeline():
    mlflow.log_params(
        {
            "version": "v2_cnn_classifier",
            "model_type": "CNNClassifier",
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with mlflow.start_span(name="load_mnist_data", span_type=SpanType.RETRIEVER) as span:
        span.set_inputs({"data_dir": DATA_DIR, "batch_size": BATCH_SIZE})
        train_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)
        span.set_outputs(
            {
                "train_examples": len(train_loader.dataset),
                "test_examples": len(test_loader.dataset),
            }
        )

    # Only line that differs from v1: CNNClassifier() instead of
    # MLPClassifier(hidden_size=HIDDEN_SIZE). No hidden_size argument
    # needed here — the CNN's channel counts are fixed inside model.py
    # for this version, since tuning them isn't today's bottleneck.
    with mlflow.start_span(name="build_cnn_model", span_type=SpanType.TOOL) as span:
        span.set_inputs({"device": str(device)})
        model = CNNClassifier().to(device)
        span.set_outputs({"parameters": sum(p.numel() for p in model.parameters())})

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_span(name="train_cnn_model", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"epochs": EPOCHS, "learning_rate": LEARNING_RATE})
        train(model, train_loader, optimizer, criterion, device, EPOCHS)
        span.set_outputs({"completed_epochs": EPOCHS})

    with mlflow.start_span(name="evaluate_cnn_model", span_type=SpanType.CHAIN) as span:
        accuracy, confusion = evaluate(model, test_loader, device)
        span.set_outputs(
            {
                "accuracy": accuracy,
                "confusion_matrix": confusion.tolist(),
            }
        )
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"\nTest accuracy: {accuracy:.4f}")
    print_confusion_matrix(confusion)

    with mlflow.start_span(name="save_cnn_checkpoint", span_type=SpanType.TOOL) as span:
        span.set_inputs({"checkpoint_path": CHECKPOINT_PATH})
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        span.set_outputs({"checkpoint_path": CHECKPOINT_PATH})
    print(f"\nModel saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
