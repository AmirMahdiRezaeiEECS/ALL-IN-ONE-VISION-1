"""
End-to-end smoke test: load a real config file the way tools/train_net.py
does, instantiate every piece, and run a couple of SimpleTrainer steps --
using the fake (network-free) dataset registered in conftest.py in place
of a real MNIST download.
"""
import os

from all_in_one_vision.checkpoint import Checkpointer
from all_in_one_vision.config import LazyConfig, instantiate
from all_in_one_vision.engine import SimpleTrainer
from all_in_one_vision.evaluation import AccuracyEvaluator, inference_on_dataset


def _load_mlp_config_with_fake_data(tmp_path):
    cfg = LazyConfig.load("configs/MNIST/mlp_baseline.py")
    # Swap in the network-free fake dataset for the test, everything else
    # about the config (model, optimizer, train settings) stays identical.
    cfg.dataloader.train.dataset_name = "fake_mnist_train"
    cfg.dataloader.test.dataset_name = "fake_mnist_test"
    cfg.dataloader.train.batch_size = 8
    cfg.dataloader.test.batch_size = 8
    cfg.train.output_dir = str(tmp_path)
    cfg.train.max_epochs = 1
    cfg.train.mlflow.enabled = False
    return cfg


def test_full_pipeline_smoke(tmp_path):
    cfg = _load_mlp_config_with_fake_data(tmp_path)

    model = instantiate(cfg.model)
    train_loader = instantiate(cfg.dataloader.train)
    test_loader = instantiate(cfg.dataloader.test)

    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)

    trainer = SimpleTrainer(model, train_loader, optimizer)
    trainer.train(start_iter=0, max_iter=3)  # a handful of steps, not a full epoch

    assert "loss_cls" in trainer.latest_losses

    evaluator = AccuracyEvaluator(num_classes=cfg.train.num_classes)
    results = inference_on_dataset(model, test_loader, evaluator)
    assert 0.0 <= results["accuracy"] <= 1.0

    checkpointer = Checkpointer(model, cfg.train.output_dir, optimizer=optimizer)
    checkpointer.save("model_test")
    assert os.path.exists(os.path.join(cfg.train.output_dir, "model_test.pth"))


def test_cli_overrides_apply():
    cfg = LazyConfig.load("configs/MNIST/mlp_baseline.py")
    cfg = LazyConfig.apply_overrides(cfg, ["train.max_epochs=99", "optimizer.lr=0.5"])
    assert cfg.train.max_epochs == 99
    assert cfg.optimizer.lr == 0.5


def test_swapping_model_via_config_is_one_line():
    """CNN baseline should load/instantiate identically to the MLP one --
    proving model swap really is config-only, no code path differs."""
    cfg = LazyConfig.load("configs/MNIST/cnn_baseline.py")
    model = instantiate(cfg.model)
    from all_in_one_vision.modeling.meta_arch import SimpleCNNClassifier

    assert isinstance(model, SimpleCNNClassifier)
