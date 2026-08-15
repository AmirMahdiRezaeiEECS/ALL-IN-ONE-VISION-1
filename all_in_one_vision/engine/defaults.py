"""
default_argument_parser
=========================
Same idea as Detectron2's engine/defaults.py::default_argument_parser:
a single, reusable CLI parser so every tool (train_net.py, a future
evaluate_net.py, ...) accepts `--config-file` plus dotted-key overrides
the same way, rather than each script inventing its own argparse setup.
"""
import argparse


def default_argument_parser(description: str = "ALL_IN_ONE_VISION_1"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config-file", required=True, help="Path to a LazyConfig .py config file"
    )
    parser.add_argument(
        "opts",
        nargs="*",
        help=(
            "Dotted-key config overrides, e.g. "
            "train.max_epochs=10 optimizer.lr=0.001"
        ),
    )
    return parser
