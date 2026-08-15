from omegaconf import OmegaConf

# NOT a LazyCall recipe -- these are plain settings read directly by
# tools/train_net.py (output paths, how often to log/checkpoint/eval,
# whether MLflow is on). Nothing here gets "instantiated" into an object.
train = OmegaConf.create(
    dict(
        output_dir="./output/default",
        max_epochs=5,
        log_period=100,
        num_classes=10,
        checkpoint_period_epochs=1,
        eval_period_epochs=1,
        mlflow=dict(
            enabled=True,
            experiment_name="all-in-one-vision",
        ),
    )
)
