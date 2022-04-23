import pickle
import os
import numpy as np
from nnreplayer.repair import NNRepair
from nnreplayer.utils import constraints_class
from nnreplayer.utils import Options

# Load Dataset
x_train, y_train, x_test, y_test = loadDataset(...)

# Load Model
model_orig = load(...)
model_orig.eval()

# Define Constraints
A = np.array([
            [-0.70710678, -0.70710678],
            [ 0.70710678, -0.70710678],
            [-0.70710678,  0.70710678],
            [ 0.70710678,  0.70710678]
            ])

b = np.array([
            [-2.31053391],
            [ 1.225     ],
            [ 1.225     ],
            [ 4.76053391]
            ])

constraint_inside = constraints_class("inside", A, b)
output_constraint_list = [constraint_inside]


# Define Layer to Repair
layer_to_repair = 3

# Define MIQP Parameters and Additional Parameters
max_weight_bound = 5
cost_weights = np.array([1.0, 1.0])
options = Options(
    "gdp.bigm",
    "gurobi",
    "python",
    "keras",
    {
        "timelimit": 3600,
        "mipgap": 0.001,
        "mipfocus": 2,
        "improvestarttime": 3300,
        "logfile": path_write
        + f"/logs/opt_log_layer{layer_to_repair}.log",
    },
)

# Intialize Repair
repair_obj = NNRepair(model_orig, "pytorch")

# Compile Repair Model
repair_obj.compile(
    x_train,
    y_train,
    layer_to_repair,
    output_constraint_list=output_constraint_list,
    cost_weights=cost_weights,
    max_weight_bound=max_weight_bound,
)

# Run Repair Process
out_model = repair_obj.repair(options)