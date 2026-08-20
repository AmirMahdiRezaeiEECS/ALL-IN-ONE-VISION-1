"""
export
========
Deployment export for trained models: TorchScript (export_torchscript)
and ONNX (export_onnx). See api.py's module docstring for why this is
much smaller than Detectron2's export/ package -- classification models
here have no custom ops or rich output structures to flatten.

Used by tools/export_model.py, the CLI entry point that loads a config +
checkpoint and calls these functions -- see that script if you want the
end-to-end "trained checkpoint -> deployable file" flow rather than
calling these functions directly.
"""
from .api import export_onnx, export_torchscript

__all__ = ["export_torchscript", "export_onnx"]
