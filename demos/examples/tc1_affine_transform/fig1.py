import os
import keras
import argparse
import numpy as np
from affine_utils import (
    plot_dataset,
    model_eval,
    original_data_loader,
    give_polys,
    give_constraints,
)
from shapely.affinity import scale


def main():
    direc = os.path.dirname(os.path.realpath(__file__))
    path_read_repair = direc + "/tc1/repair_net"
    path_read_orig_model = direc + "/tc1/original_net"

    model_orig = keras.models.load_model(path_read_orig_model + "/model")
    model_repair_1 = keras.models.load_model(
        path_read_repair + "/model_layer_1"
    )
    model_repair_2 = keras.models.load_model(
        path_read_repair + "/model_layer_2"
    )
    model_repair_3 = keras.models.load_model(
        path_read_repair + "/model_layer_3"
    )

    pass


if __name__ == "__main__":
    main()
    pass
