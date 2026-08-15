# Testing Guide

## Philosophy

Per this project's development approach ("almost test-oriented"), tests
here exist to make two things safe: **refactoring** (change how
something works internally without breaking what depends on it) and
**regression prevention** (catch a future change that breaks something
that used to work). They are not attempting to be exhaustive — for an
MVP, a handful of well-chosen tests that exercise the *architecture*
(does the config system really compose correctly? does swapping models
really require zero other changes?) are worth more than a large number
of tests checking numerical minutiae.

## Why the tests don't need a real MNIST download

`tests/conftest.py` registers two extra entries in the `DatasetCatalog`
— `"fake_mnist_train"` / `"fake_mnist_test"` — backed by a tiny in-memory
dataset of random tensors shaped exactly like MNIST images
(`(1, 28, 28)`) with random labels in `[0, 10)`. This exists purely so
tests can exercise the *real* pipeline (real catalog lookup, real
`DataLoader`, real model, real training step) without needing network
access or waiting for a download — the point of the tests is proving the
architecture wires together correctly, not verifying MNIST's actual
pixel values.

```python
# tests/conftest.py, simplified
class _FakeMNIST(Dataset):
    def __init__(self, n=32, transform=None):
        self.images = torch.rand(n, 1, 28, 28)
        self.labels = torch.randint(0, 10, (n,))
    ...

DatasetCatalog.register("fake_mnist_train", lambda transform: _FakeMNIST(64, transform))
```

This registration runs automatically (as an import side effect) whenever
any test file is collected by `pytest`, because `conftest.py` is
special — `pytest` always imports it before running tests in that
directory.

## What each test file proves

| File | What it proves |
|---|---|
| `test_instantiate.py` | `LazyCall` really is inert until `instantiate()` runs; `instantiate()` correctly recurses into nested recipes |
| `test_catalog.py` | MNIST really gets registered on import; `build_loader` correctly turns a catalog name into a working `DataLoader`; looking up an unregistered name raises a clear error |
| `test_meta_arch.py` | Both `SimpleMLPClassifier` and `SimpleCNNClassifier` correctly follow the shared forward contract (loss dict in train mode, logits in eval mode) — the exact property that lets `engine/` and `evaluation/` stay model-agnostic |
| `test_config_pipeline.py` | The **entire** pipeline — load a real config file, instantiate every piece, run real training steps, evaluate, checkpoint — works end to end; also proves CLI overrides actually take effect, and that swapping `mlp_baseline.py` for `cnn_baseline.py` really is a one-line, code-free change |

`test_config_pipeline.py::test_full_pipeline_smoke` is the most
important single test in the suite — it's the closest thing to "does the
whole architecture actually work" in one function. If you're not sure
whether a change broke something structural, run this test first.

## Running the tests

```bash
pip install pytest
PYTHONPATH=. pytest tests/ -v
```

(`PYTHONPATH=.` is needed so `all_in_one_vision` and `configs` resolve as
importable packages when pytest is run from the project root without an
editable install — the same reason `tools/train_net.py` has its
`sys.path.insert(...)` line.)

## Adding a test for something you just added

If you followed `04_extending_the_project.md` to add a new dataset or
model, consider adding:
- **New dataset:** a one-line addition to `test_catalog.py`'s
  registration check (`assert "your_dataset_train" in names`), and
  optionally a `build_loader` smoke test like the existing
  `fake_mnist_train` one (no need to hit the real network — register a
  fake version in `conftest.py` if you want a full loader test).
- **New model:** add it to the `_check_train_eval_contract(...)` calls
  in `test_meta_arch.py` — this one function verifies the forward
  contract for any model in a couple of lines:
  ```python
  def test_deep_mlp_contract():
      _check_train_eval_contract(DeepMLPClassifier(hidden_dims=(64, 32)))
  ```

## What's deliberately NOT tested (yet)

- Actual model accuracy/convergence on real MNIST (that's a training
  outcome to observe, not a unit-testable property — an MVP milestone
  like "MLP reaches ~97% test accuracy in 5 epochs" belongs in a
  README/experiment log, not an assertion in `pytest`).
- MLflow logging correctness beyond "does it run without error" (already
  manually verified once against a real local MLflow store during
  development — see the project's `README.md` for that note). Could
  become a real test later by asserting on the on-disk run artifacts
  MLflow writes, if that becomes a recurring source of bugs.
- The `backbone/`/`heads/` placeholder packages — nothing to test yet,
  since nothing is implemented there.

➡️ Next: [`07_faq_and_design_deviations.md`](./07_faq_and_design_deviations.md)
