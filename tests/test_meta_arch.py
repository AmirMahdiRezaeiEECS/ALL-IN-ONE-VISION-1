import torch

from all_in_one_vision.modeling.meta_arch import SimpleMLPClassifier, SimpleCNNClassifier


def _check_train_eval_contract(model):
    images = torch.rand(4, 1, 28, 28)
    targets = torch.randint(0, 10, (4,))

    model.train()
    out = model(images, targets)
    assert isinstance(out, dict)
    assert "loss_cls" in out
    assert out["loss_cls"].dim() == 0  # scalar loss

    model.eval()
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (4, 10)


def test_mlp_contract():
    _check_train_eval_contract(SimpleMLPClassifier(in_features=28 * 28, hidden_dim=32, num_classes=10))


def test_cnn_contract():
    _check_train_eval_contract(SimpleCNNClassifier(in_channels=1, num_classes=10))

def test_cnn_contract_cifar10_shape():
    images = torch.rand(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    model = SimpleCNNClassifier(in_channels=3, num_classes=10, image_size=32)

    model.train()
    out = model(images, targets)
    assert out["loss_cls"].dim() == 0

    model.eval()
    with torch.no_grad():
        logits = model(images)
    assert logits.shape == (4, 10)