from omegaconf import OmegaConf
from torchvision import transforms as T

from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.data import build_loader

# Standard MNIST normalization constants (dataset mean/std) -- the same
# values torchvision's own MNIST examples use.
transform = L(T.Compose)(
    transforms=[
        L(T.ToTensor)(),
        L(T.Normalize)(mean=[0.1307], std=[0.3081]),
    ]
)

dataloader = OmegaConf.create()
dataloader.train = L(build_loader)(
    dataset_name="mnist_train",
    transform=transform,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
dataloader.test = L(build_loader)(
    dataset_name="mnist_test",
    transform=transform,
    batch_size=1000,
    shuffle=False,
    num_workers=0,
)
