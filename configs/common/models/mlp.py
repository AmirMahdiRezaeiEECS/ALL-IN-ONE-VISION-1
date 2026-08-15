from all_in_one_vision.config import LazyCall as L
from all_in_one_vision.modeling.meta_arch import SimpleMLPClassifier

model = L(SimpleMLPClassifier)(
    in_features=28 * 28,
    hidden_dim=128,
    num_classes=10,
)
