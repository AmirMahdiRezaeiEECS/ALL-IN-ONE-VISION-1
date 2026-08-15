"""
Optimizer config
==================
Describes the optimizer (plain SGD with momentum) as a LazyCall recipe --
with one twist explained below. Full deep-dive:
docs/02_config_system_deep_dive.md ("The optimizer special case").

THE `params=L(get_default_optimizer_params)(model=None)` LINE:
an optimizer's constructor needs `model.parameters()`, but at the point
this config file is imported, no model object exists yet -- only a
recipe for one (see configs/common/models/mlp.py). So instead of passing
a real value here, we pass ANOTHER recipe, whose `model` argument starts
out as a placeholder (`None`).

tools/train_net.py builds the real model first, then does:
    cfg.optimizer.params.model = model      # inject the real, already-built model
    optimizer = instantiate(cfg.optimizer)    # NOW everything resolves correctly

If you're not modifying that runtime-linking step, you don't need to
think about this again -- just treat `optimizer.lr` and
`optimizer.momentum` below as the two knobs you'll actually want to turn
day to day (e.g. via `optimizer.lr=0.005` on the command line).
"""
import torch

from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.solver import get_default_optimizer_params

optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(model=None),  # filled in at runtime, see above
    lr=0.01,        # learning rate
    momentum=0.9,   # SGD momentum -- helps convergence speed/stability
)
