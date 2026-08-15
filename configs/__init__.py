"""
configs
=========
Pure DATA, not library code -- every file under this package is meant to
be read, copied, and edited by you, not imported as a reusable API.

This is an ordinary Python package (every subfolder has __init__.py)
specifically so config files can `import` each other with plain,
absolute imports (see docs/07_faq_and_design_deviations.md for why this
project does it this way instead of Detectron2's relative-import
patching).

    common/   -- shared building blocks: one file per model/dataset/
                  optimizer/training-settings piece, meant to be
                  imported and composed by experiment configs below.
    MNIST/    -- experiment configs: each file composes a few common/
                  building blocks into one complete, runnable experiment.

See docs/02_config_system_deep_dive.md for exactly how these files turn
into real training runs.
"""
