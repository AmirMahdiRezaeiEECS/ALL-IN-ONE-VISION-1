from configs.common.models.mlp import model
from configs.common.data.fashion_mnist import dataloader
from configs.common.optim import optimizer
from configs.common.train import train

train.output_dir = "./output/fashion_mnist_mlp_v1"
train.max_epochs = 5
train.mlflow.experiment_name = "all-in-one-vision-fashion-mnist"
