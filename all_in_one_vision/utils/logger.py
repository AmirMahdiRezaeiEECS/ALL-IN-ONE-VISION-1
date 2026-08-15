"""
setup_logger
=============
Reuses Python's standard `logging` module -- no custom logging framework.
This is intentionally the simplest thing that works; Detectron2's version
adds color formatting and distributed-training rank filtering, neither of
which matters yet on a single-machine classification MVP.
"""
import logging
import sys


def setup_logger(name: str = "all_in_one_vision", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured, avoid duplicate handlers
    logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
