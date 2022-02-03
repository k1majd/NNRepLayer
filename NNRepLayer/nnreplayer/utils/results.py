from dataclasses import dataclass
from typing import Any

@dataclass
class Results:
    weights: Any
    bias: Any
    new_weight: Any
    new_bias: Any
    new_model: Any
    MSE_original_nn_train: Any
    MSE_new_nn_train: Any
    weight_error: Any
    bias_error: Any