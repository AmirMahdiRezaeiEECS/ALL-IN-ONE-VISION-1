# backbone/

Placeholder package. Not used in v1 -- SimpleMLPClassifier and
SimpleCNNClassifier are each small enough to be a single self-contained
meta-arch (like Detectron2's simplest meta-archs).

This becomes relevant once a model needs a *swappable* feature extractor
(e.g. a ResNet backbone reused across several future meta-archs). At that
point, mirror Detectron2's pattern: an `fvcore.common.registry.Registry`
here (`BACKBONE_REGISTRY`), so `cfg` can pick a backbone by name, and
config files stay declarative.
