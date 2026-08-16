"""
Data config: CIFAR-10
========================
Mirrors configs/common/data/mnist.py's structure exactly -- see that
file's docstring for the full explanation of the transform/dataloader
split and why this file only refers to the dataset by NAME
("cifar10_train"/"cifar10_test"), never a class or path directly.

THE NORMALIZE VALUES (mean=[0.4914, 0.4822, 0.4465],
std=[0.2470, 0.2435, 0.2616]) are CIFAR-10's actual, well-known
per-channel (R, G, B) pixel statistics computed over the training set --
the same constants used across essentially every CIFAR-10 tutorial/paper,
analogous to MNIST's mean=0.1307/std=0.3081 in mnist.py.
"""
from omegaconf import OmegaConf
from torchvision import transforms as T

from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.data import build_loader

transform = L(T.Compose)(
    transforms=[
        L(T.ToTensor)(),
        L(T.Normalize)(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ]
)

dataloader = OmegaConf.create()
dataloader.train = L(build_loader)(
    dataset_name="cifar10_train",
    transform=transform,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
dataloader.test = L(build_loader)(
    dataset_name="cifar10_test",
    transform=transform,
    batch_size=1000,
    shuffle=False,
    num_workers=0,
)