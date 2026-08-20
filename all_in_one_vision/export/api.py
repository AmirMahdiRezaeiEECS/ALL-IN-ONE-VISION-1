"""
Export utilities
==================
Exports a trained classification meta-arch to deployment formats
(TorchScript, ONNX) for downstream consumers outside this training
codebase.

WHY THIS IS SIMPLE (compared to Detectron2's export/ package): every
meta-arch here follows the shared forward contract (dict of losses in
train mode, raw logits in eval mode -- see modeling/meta_arch/mlp.py's
docstring), and the eval-mode forward path has no dynamic control flow,
no custom ops, and no rich output structures (Instances/Boxes) that need
flattening. That means plain `torch.jit.trace` and `torch.onnx.export`
work directly on the model as-is -- none of Detectron2's export/
machinery (Caffe2 c10 ops, TracingAdapter/Schema flattening, ONNX opset
patching) is needed.

REUSE-FIRST: this module is a thin wrapper around torch.jit / torch.onnx
-- it does not reimplement tracing or graph export.

Both functions expect:
  - `model` already in eval mode (asserted defensively below).
  - `example_input` a single image-batch tensor of the exact shape the
    exported model will be served with. Both TorchScript tracing and (by
    default here) ONNX export bake in the traced shape; use
    `dynamic_batch=True` (the default for export_onnx) to keep the batch
    dimension flexible, since that's the one axis downstream servers
    almost always need to vary.
"""
import torch


def export_torchscript(model, example_input, out_path):
    """
    Trace `model` and save it as a TorchScript file.

    Args:
        model: an nn.Module already in eval mode, following this
            project's forward contract (forward(images) -> logits).
        example_input: a tensor, shape (batch, channels, height, width),
            used to trace the graph.
        out_path: file path to write the traced module to (e.g. ending
            in ".pt").

    Returns:
        out_path, for convenient chaining/logging.
    """
    assert not model.training, "export_torchscript expects a model in eval mode"
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    traced.save(out_path)
    return out_path


def export_onnx(model, example_input, out_path, opset_version=18, dynamic_batch=True):
    """
    Export `model` to ONNX format.

    Args:
        model: an nn.Module already in eval mode.
        example_input: a tensor used to trace the graph (see
            export_torchscript's docstring).
        out_path: file path to write the .onnx file to.
        opset_version: ONNX opset to target. 18 is a safe, broadly
            supported default as of this writing; override if a
            downstream runtime needs a different one.
        dynamic_batch: if True (default), the exported graph accepts any
            batch size at inference time rather than being locked to
            example_input's batch size -- the one axis downstream
            serving code almost always needs to vary. Uses
            `torch.export.Dim` / `dynamic_shapes` (not the older
            `dynamic_axes` dict), which is the non-deprecated way to
            express this on current torch ONNX-export internals.

    Returns:
        out_path, for convenient chaining/logging.
    """
    assert not model.training, "export_onnx expects a model in eval mode"
    dynamic_shapes = None
    if dynamic_batch:
        from torch.export import Dim

        dynamic_shapes = ({0: Dim("batch")},)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_input,
            out_path,
            input_names=["images"],
            output_names=["logits"],
            dynamic_shapes=dynamic_shapes,
            opset_version=opset_version,
        )
    return out_path
