# v1 — MNIST Classifier (MLP baseline)

## Scope (what THIS version delivers)
- Load MNIST via `torchvision.datasets` (reuse-first: no manual data parsing).
- A simple fully-connected MLP (no convolutions yet — that's v2).
- A complete, working train -> evaluate -> save loop.
- Clear module boundaries so v2 can swap in a CNN by only touching `model.py`.

## Explicitly deferred (NOT in this version)
- CNN / convolutional architectures.
- Data augmentation.
- Hyperparameter tuning / experiment tracking.
- Any robotics-specific CV. This version is pure "practice from zero."

## Structure
```
v1_mnist_classifier/
├── README.md
├── requirements.txt
├── main.py              # entry point: ties everything together
├── src/
│   ├── dataset.py       # data loading + preprocessing
│   ├── model.py         # the MLP definition (only file v2 needs to change)
│   ├── train.py         # training loop
│   └── evaluate.py      # accuracy + confusion matrix
└── saved_models/        # trained checkpoints land here (empty until you run it)
```

## How to run
```bash
cd v1_mnist_classifier
pip install -r requirements.txt
python main.py
```

First run will auto-download MNIST (~11MB) into a local `./data` folder via
torchvision. Trains on CPU in a couple of minutes; use a GPU machine if you
want it faster, but it's not required for MNIST.

## Design notes
- `model.py` is isolated on purpose: architecture is a "Domain concept" of
  its own, so version 2 (CNN) only replaces this one file — everything else
  (data loading, training loop, evaluation) stays the same.
- Config values (batch size, learning rate, epochs) are kept as constants
  at the top of `main.py` for now — no config file yet, since that would be
  over-engineering for a single-script MVP. If v2+ grows more complex,
  a config file becomes justified then, not now.
