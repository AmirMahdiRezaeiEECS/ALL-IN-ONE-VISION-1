"""
LazyCall ("L")
==============

WHY THIS EXISTS
----------------
A normal Python config would import a model class and construct it directly:

    model = SimpleMLPClassifier(hidden_dim=128)

The problem: this *runs* the constructor the moment the config file is
imported. That's fine for simple scripts, but it means configs can't be
composed, overridden from the command line, or serialized/inspected before
anything is actually built.

LazyCall solves this by wrapping the call so it is *recorded*, not
*executed*:

    model = L(SimpleMLPClassifier)(hidden_dim=128)

`model` here is NOT a SimpleMLPClassifier instance. It's a small
dict-like recipe: {"_target_": "all_in_one_vision...SimpleMLPClassifier",
"hidden_dim": 128}. Nothing is constructed yet.

Only when `instantiate(model)` is called (see instantiate.py) does the
recipe actually get turned into a real object. Because the recipe is just
plain data, it can be:
  - printed / inspected before training starts
  - overridden field-by-field from the CLI (e.g. `model.hidden_dim=256`)
  - nested arbitrarily (a recipe's kwargs can themselves be recipes, e.g.
    an optimizer recipe whose `params` kwarg is itself a recipe)

This is the same mechanism Detectron2's LazyConfig system uses for its
Python-based configs (configs/common/*.py + config/lazy.py upstream).
"""
from omegaconf import DictConfig


def _target_to_string(target):
    """
    Convert a class/function object into an "import path" string, e.g.
    the class `SimpleMLPClassifier` defined in
    `all_in_one_vision.modeling.meta_arch.mlp` becomes the string
    "all_in_one_vision.modeling.meta_arch.mlp.SimpleMLPClassifier".

    We store the STRING (not the live object) inside the config so that
    the config remains plain, serializable data -- resolving the string
    back into a real class only happens later, inside instantiate().
    """
    module = target.__module__
    qualname = target.__qualname__
    return f"{module}.{qualname}"


class LazyCall:
    """
    Wrap a callable `target` (a class or a function). Calling the wrapped
    object with keyword arguments produces a declarative recipe (an
    OmegaConf DictConfig) rather than a real instance.

    Example
    -------
        from all_in_one_vision.config import LazyCall as L
        from all_in_one_vision.modeling.meta_arch.mlp import SimpleMLPClassifier

        model = L(SimpleMLPClassifier)(hidden_dim=128, num_classes=10)
        # model == {"_target_": "....SimpleMLPClassifier",
        #           "hidden_dim": 128, "num_classes": 10}
    """

    def __init__(self, target):
        if not (isinstance(target, str) or callable(target)):
            raise TypeError(
                f"LazyCall target must be a callable or an import path string, got {target!r}"
            )
        self._target = target

    def __call__(self, **kwargs):
        target = self._target
        target_str = target if isinstance(target, str) else _target_to_string(target)
        kwargs["_target_"] = target_str
        # allow_objects=True: lets values be arbitrary Python objects
        # (not just primitives). We need this because at *runtime* we
        # sometimes inject already-built objects into a recipe before
        # instantiating it -- e.g. plugging a live `model` into the
        # optimizer recipe's `params` field (see tools/train_net.py).
        return DictConfig(content=kwargs, flags={"allow_objects": True})
