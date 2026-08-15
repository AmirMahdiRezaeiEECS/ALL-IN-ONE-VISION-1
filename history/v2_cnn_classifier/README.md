# v2 — MNIST Classifier (CNN upgrade)

## Scope (what THIS version delivers)
- Same MNIST data, same train -> evaluate -> save loop as v1.
- Replace the MLP with a small CNN (2 conv blocks + 1 linear layer).
- Prove out the architecture-swap boundary v1 was designed around:
  **only `src/model.py` changes.** `dataset.py`, `train.py`, and
  `evaluate.py` are byte-for-byte copies of v1.

## Explicitly deferred (NOT in this version)
- Data augmentation.
- Batch normalization, dropout, or any regularization.
- Deeper/wider architectures, hyperparameter tuning, experiment tracking.
- Moving off MNIST onto real-world or robotics-specific data (that's the
  next bottleneck to tackle after this version, per "data > models").

## Structure
```
v2_cnn_classifier/
├── README.md
├── requirements.txt      # unchanged from v1
├── main.py               # unchanged except: imports CNNClassifier
├── src/
│   ├── dataset.py        # UNCHANGED copy of v1
│   ├── model.py          # NEW: small CNN (the only real change)
│   ├── train.py          # UNCHANGED copy of v1
│   └── evaluate.py        # UNCHANGED copy of v1
└── saved_models/         # trained checkpoints land here
```

## How to run
```bash
cd v2_cnn_classifier
pip install -r requirements.txt
python main.py
```
Reuses the same `./data` MNIST cache format as v1 (re-downloads if not
present in this folder).

## Design notes
- This version is deliberately a **validation step**, not a capability
  leap: same dataset, same loop, only the architecture changes. The
  point is to confirm v1's module boundaries actually hold up under a
  real change, before trusting them for bigger future swaps.
- Expect meaningfully higher test accuracy than v1's MLP (CNNs are
  well-suited to image data in a way flattened MLPs aren't), but that's
  a nice side effect here, not the primary goal of this version.
- Next bottleneck candidate (not started): MNIST itself is a very easy,
  clean, non-robotics dataset. Once this version is validated, the
  likely next step is moving toward real-world / robotics-relevant
  image data, per the "data is more important than models" principle.
