"""
all_in_one_vision.config
==========================
The declarative config system: everything about "what to build" (model,
data pipeline, optimizer) is expressed as plain, inspectable Python data
until the last possible moment, then turned into real objects in one
step.

The three pieces, and the order you'll typically meet them in:

    LazyCall (aliased "L")  -- wrap a class/function + kwargs into a
                                declarative recipe instead of calling it.
    instantiate()            -- walk a recipe (or tree of recipes) and
                                actually build the real objects.
    LazyConfig                -- load a whole config .py file (which is
                                just a bag of such recipes) into one
                                OmegaConf tree, and apply CLI overrides.

See docs/03_config_system.md for a full walkthrough with examples, and
lazy_call.py / instantiate.py / lazy.py in this package for the
implementation-level "why" of each piece.
"""
from .lazy_call import LazyCall
from .instantiate import instantiate
from .lazy import LazyConfig

__all__ = ["LazyCall", "instantiate", "LazyConfig"]
