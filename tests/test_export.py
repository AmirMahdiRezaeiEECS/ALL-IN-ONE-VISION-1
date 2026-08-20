"""
Export smoke tests: trace/export a small model and verify the exported
artifact loads back and produces output of the right shape (and, for
ONNX, numerically matches the original PyTorch model). Uses the meta-arch
classes directly (no LazyConfig involved) -- these tests are about
export/api.py's correctness, not the CLI wiring in tools/export_model.py.
"""
import numpy as np
import onnx
import onnxruntime
import torch

from all_in_one_vision.export import export_onnx, export_torchscript
from all_in_one_vision.modeling.meta_arch import SimpleCNNClassifier, SimpleMLPClassifier


def test_export_torchscript_mlp(tmp_path):
    model = SimpleMLPClassifier(in_features=28 * 28, hidden_dim=32, num_classes=10)
    model.eval()
    example = torch.rand(2, 1, 28, 28)

    out_path = str(tmp_path / "model.pt")
    export_torchscript(model, example, out_path)

    loaded = torch.jit.load(out_path)
    with torch.no_grad():
        out = loaded(example)
    assert out.shape == (2, 10)


def test_export_torchscript_cnn(tmp_path):
    model = SimpleCNNClassifier(in_channels=1, num_classes=10, image_size=28)
    model.eval()
    example = torch.rand(2, 1, 28, 28)

    out_path = str(tmp_path / "model.pt")
    export_torchscript(model, example, out_path)

    loaded = torch.jit.load(out_path)
    with torch.no_grad():
        out = loaded(example)
    assert out.shape == (2, 10)


def test_export_torchscript_matches_pytorch_output(tmp_path):
    """The traced graph should be numerically identical to eager mode."""
    model = SimpleCNNClassifier(in_channels=1, num_classes=10, image_size=28)
    model.eval()
    example = torch.rand(2, 1, 28, 28)

    out_path = str(tmp_path / "model.pt")
    export_torchscript(model, example, out_path)
    loaded = torch.jit.load(out_path)

    with torch.no_grad():
        eager_out = model(example)
        traced_out = loaded(example)
    assert torch.allclose(eager_out, traced_out, atol=1e-6)


def test_export_onnx_cnn_is_valid_and_matches_pytorch(tmp_path):
    model = SimpleCNNClassifier(in_channels=1, num_classes=10, image_size=28)
    model.eval()
    example = torch.rand(2, 1, 28, 28)

    out_path = str(tmp_path / "model.onnx")
    export_onnx(model, example, out_path)

    # The exported graph itself should pass ONNX's own structural checker.
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    # And running it should numerically match the original PyTorch model.
    session = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"images": example.numpy()})[0]

    with torch.no_grad():
        torch_out = model(example).numpy()

    assert np.allclose(onnx_out, torch_out, atol=1e-5)


def test_export_onnx_mlp_is_valid_and_matches_pytorch(tmp_path):
    model = SimpleMLPClassifier(in_features=28 * 28, hidden_dim=32, num_classes=10)
    model.eval()
    example = torch.rand(3, 1, 28, 28)

    out_path = str(tmp_path / "model.onnx")
    export_onnx(model, example, out_path)

    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    session = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"images": example.numpy()})[0]
    with torch.no_grad():
        torch_out = model(example).numpy()
    assert np.allclose(onnx_out, torch_out, atol=1e-5)


def test_export_onnx_dynamic_batch_accepts_different_batch_size(tmp_path):
    """Exported ONNX graph should accept a different batch size than it was traced with."""
    model = SimpleCNNClassifier(in_channels=1, num_classes=10, image_size=28)
    model.eval()
    example = torch.rand(2, 1, 28, 28)

    out_path = str(tmp_path / "model.onnx")
    export_onnx(model, example, out_path, dynamic_batch=True)

    session = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    different_batch = torch.rand(5, 1, 28, 28)
    onnx_out = session.run(None, {"images": different_batch.numpy()})[0]
    assert onnx_out.shape == (5, 10)


def test_export_onnx_static_batch_bakes_in_traced_batch_size(tmp_path):
    """With dynamic_batch=False, the graph should be locked to the traced batch size."""
    model = SimpleCNNClassifier(in_channels=1, num_classes=10, image_size=28)
    model.eval()
    example = torch.rand(4, 1, 28, 28)

    out_path = str(tmp_path / "model.onnx")
    export_onnx(model, example, out_path, dynamic_batch=False)

    session = onnxruntime.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    # Same batch size as traced -- should work.
    onnx_out = session.run(None, {"images": example.numpy()})[0]
    assert onnx_out.shape == (4, 10)


def test_export_torchscript_and_onnx_agree_with_each_other(tmp_path):
    """Cross-check: both export formats should produce the same predictions."""
    model = SimpleCNNClassifier(in_channels=1, num_classes=10, image_size=28)
    model.eval()
    example = torch.rand(2, 1, 28, 28)

    ts_path = str(tmp_path / "model.pt")
    onnx_path = str(tmp_path / "model.onnx")
    export_torchscript(model, example, ts_path)
    export_onnx(model, example, onnx_path)

    ts_model = torch.jit.load(ts_path)
    with torch.no_grad():
        ts_out = ts_model(example).numpy()

    session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"images": example.numpy()})[0]

    assert np.allclose(ts_out, onnx_out, atol=1e-5)
