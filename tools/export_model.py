"""
export_model.py
=================
Export a trained model (built from a LazyConfig experiment config, with
weights loaded from a checkpoint) to TorchScript and/or ONNX for
downstream consumers outside this training codebase.

Mirrors tools/train_net.py's own "load config -> instantiate" shape --
see docs/03_training_workflow_walkthrough.md for that pattern in detail.
This script does NOT touch engine/, evaluation/, or train_net.py itself;
it only reads a config + checkpoint and calls all_in_one_vision/export/.

Usage:
    python tools/export_model.py \
        --config-file configs/MNIST/cnn_baseline.py \
        --checkpoint output/mnist_cnn_v1/model_final.pth \
        --format torchscript onnx \
        --output-dir output/mnist_cnn_v1/export

    # CIFAR-10 example (3 channels, 32x32) -- shape is read from the
    # config automatically, no extra flags needed:
    python tools/export_model.py \
        --config-file configs/CIFAR-10/cnn_baseline.py \
        --checkpoint output/cifar10_cnn_v1/model_final.pth \
        --format onnx \
        --output-dir output/cifar10_cnn_v1/export
"""
import argparse
import os
import sys

# Allow running as `python tools/export_model.py` from the project root
# without an editable install -- same convenience line as train_net.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from all_in_one_vision.checkpoint import Checkpointer
from all_in_one_vision.config import LazyConfig, instantiate
from all_in_one_vision.export import export_onnx, export_torchscript
from all_in_one_vision.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained model to TorchScript/ONNX")
    parser.add_argument(
        "--config-file", required=True,
        help="Path to the LazyConfig experiment file the model was trained with",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a checkpoint file saved by Checkpointer (e.g. model_final.pth)",
    )
    parser.add_argument(
        "--format", nargs="+", choices=["torchscript", "onnx"], default=["torchscript", "onnx"],
        help="Which format(s) to export (default: both)",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write exported files to")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size baked into the traced example input (default: 1). Irrelevant for "
             "ONNX output if the dynamic batch axis is left enabled (the default).",
    )
    parser.add_argument(
        "--static-batch", action="store_true",
        help="For ONNX export, lock the graph to --batch-size instead of allowing a dynamic "
             "batch dimension. Off by default -- most downstream servers want a variable batch.",
    )
    parser.add_argument("--opset-version", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "opts", nargs="*",
        help="Dotted-key config overrides, e.g. model.in_channels=3 model.image_size=32 "
             "(only needed if the config file itself doesn't already set the right shape)",
    )
    return parser.parse_args()


def _infer_example_input(cfg, batch_size):
    """
    Build a dummy input tensor matching the model's expected shape,
    read directly off the loaded config -- so this script never needs a
    per-dataset special case (MNIST vs CIFAR-10 vs any future dataset
    all fall out of the same two fields).

    - CNN configs expose `in_channels` and `image_size` directly (see
      configs/common/models/cnn.py) -- use those when present.
    - MLP configs expose `in_features` instead (flattened pixel count);
      since MLP inputs in this project are always square single- or
      multi-channel images, we only need image_size, not channels, to
      reconstruct the right (C, H, W) shape for tracing -- MLP is
      currently MNIST-only in this project (see
      docs/04_extending_the_project.md), so `in_channels=1` is a safe
      default there.
    """
    model_cfg = cfg.model
    if "image_size" in model_cfg and "in_channels" in model_cfg:
        c, s = model_cfg.in_channels, model_cfg.image_size
        return torch.rand(batch_size, c, s, s)

    if "in_features" in model_cfg:
        side = int(round(model_cfg.in_features ** 0.5))
        assert side * side == model_cfg.in_features, (
            f"Could not infer a square image shape from in_features={model_cfg.in_features}; "
            "pass an explicit shape via config overrides or extend _infer_example_input()."
        )
        return torch.rand(batch_size, 1, side, side)

    raise ValueError(
        "Could not infer example input shape from cfg.model "
        f"(no image_size/in_channels or in_features field found: {dict(model_cfg)!r})"
    )


def main():
    args = parse_args()
    logger = setup_logger()

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    model = instantiate(cfg.model)
    Checkpointer(model).load(args.checkpoint)
    model.eval()
    logger.info(f"Loaded weights from {args.checkpoint}")

    example_input = _infer_example_input(cfg, args.batch_size)
    logger.info(f"Using example input shape {tuple(example_input.shape)} for tracing")

    os.makedirs(args.output_dir, exist_ok=True)

    if "torchscript" in args.format:
        out_path = os.path.join(args.output_dir, "model.torchscript.pt")
        export_torchscript(model, example_input, out_path)
        logger.info(f"Exported TorchScript model to {out_path}")

    if "onnx" in args.format:
        out_path = os.path.join(args.output_dir, "model.onnx")
        export_onnx(
            model, example_input, out_path,
            opset_version=args.opset_version,
            dynamic_batch=not args.static_batch,
        )
        logger.info(f"Exported ONNX model to {out_path}")


if __name__ == "__main__":
    main()
