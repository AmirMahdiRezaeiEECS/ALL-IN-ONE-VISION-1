# transforms/

Placeholder package for classification-specific transform helpers.

For v1, transforms are expressed directly in config files as
`LazyCall(torchvision.transforms.Compose)(...)` -- torchvision's transforms
already cover normalization/augmentation, so there's nothing to reimplement
yet. This package exists so that if/when we need custom transforms (e.g.
robotics-specific augmentations later), they have an obvious, established
home without restructuring `data/`.
