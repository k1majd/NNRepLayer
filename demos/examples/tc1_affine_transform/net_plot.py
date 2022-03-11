import os
from tensorflow import keras
from affine_utils import (
    label_output_inside,
    plot_history,
    plot_dataset,
    model_eval,
    original_data_loader,
    give_polys,
)
import numpy as np

path_read = os.path.dirname(os.path.realpath(__file__)) + "/tc1/repair_net"

model_orig = keras.models.load_model(path_read + "/model_1")
new_weights = model_orig.get_weights()
model_orig.set_weights(new_weights)

# extract network architecture
architecture = []
for lnum, lay in enumerate(model_orig.layers):
    architecture.append(lay.input.shape[1])
    if lnum == len(model_orig.layers) - 1:
        architecture.append(lay.output.shape[1])
# load dataset and constraints
x_train, y_train, x_test, y_test = original_data_loader()
poly_orig, poly_trans, poly_const = give_polys()


plot_dataset(
    [poly_orig, poly_trans, poly_const],
    [y_train, model_orig.predict(x_train)],
    label="training",
)
plot_dataset(
    [poly_orig, poly_trans, poly_const],
    [y_test, model_orig.predict(x_test)],
    label="testing",
)
