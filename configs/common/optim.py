import torch

from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.solver import get_default_optimizer_params

# `model=None` here is a placeholder. tools/train_net.py sets
# `cfg.optimizer.params.model = model` (the REAL, already-built model)
# right before calling instantiate(cfg.optimizer) -- see solver/build.py
# for why this two-step dance is necessary.
optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(model=None),
    lr=0.01,
    momentum=0.9,
)
