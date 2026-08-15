"""
get_default_optimizer_params
==============================
Optimizers need `model.parameters()`, but at config-authoring time no
model object exists yet -- only a LazyCall *recipe* for one. Detectron2
solves this by making the optimizer's `params` argument itself a LazyCall
recipe (see configs/common/optim.py) whose `model` field starts out as
`None` and gets filled in with the real, already-built model right before
`instantiate(cfg.optimizer)` is called (see tools/train_net.py):

    cfg.optimizer.params.model = model      # inject the real model
    optimizer = instantiate(cfg.optimizer)   # NOW params= gets resolved

This function is intentionally minimal for v1 -- just `model.parameters()`.
Detectron2's real version supports per-parameter-group overrides (e.g.
different LR/weight-decay for biases vs. weights). We're not there yet;
this is the obvious place to add that once it's actually needed.
"""


def get_default_optimizer_params(model):
    return model.parameters()
