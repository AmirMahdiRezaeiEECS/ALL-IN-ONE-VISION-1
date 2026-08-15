"""
all_in_one_vision
====================
A small, Detectron2-inspired computer vision library for image
classification. New here? Start with docs/00_start_here.md at the
project root -- it's a guided tour, not just a reference.

Package map (see docs/01_architecture_and_concepts.md for the "why"
behind this layout):
    config/      -- the declarative config system (LazyCall, instantiate, LazyConfig)
    data/        -- dataset registration (DatasetCatalog) and DataLoader building
    modeling/    -- models ("meta-archs"): SimpleMLPClassifier, SimpleCNNClassifier
    solver/      -- optimizer-construction helpers
    engine/      -- the training loop (SimpleTrainer) and its hooks
    evaluation/  -- the DatasetEvaluator interface and AccuracyEvaluator
    checkpoint/  -- model/optimizer checkpoint saving & loading
    utils/       -- small reused-not-reinvented helpers (logging, env info)
"""
__version__ = "1.0.0"
