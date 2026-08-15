from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.modeling.meta_arch import SimpleCNNClassifier

model = L(SimpleCNNClassifier)(
    in_channels=1,
    num_classes=10,
)
