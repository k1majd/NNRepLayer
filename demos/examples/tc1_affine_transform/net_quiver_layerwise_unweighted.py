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
    plot_quiver,
    net_meshgrid_prediction,
    transform_mesh,
)
from shapely.affinity import scale


def main():
    """_summary_"""
    direc = os.path.dirname(os.path.realpath(__file__))
    path_read_repair = direc + "/tc1/repair_net"
    path_read_orig_model = direc + "/tc1/original_net"
    path_write = direc + "/tc1/figures/figures_unweighted"
    if not os.path.exists(path_write):
        os.makedirs(path_write)

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

    # bound polys
    poly_orig, poly_trans, poly_const = give_polys()

    # meshgri on input
    n_x = 21
    X = np.linspace(1, 4, n_x)
    Y = np.linspace(1, 4, n_x)
    xx, yy = np.meshgrid(X, Y)

    xx_trans, yy_trans = transform_mesh(xx, yy)

    # original model
    x_out, y_out = net_meshgrid_prediction(xx, yy, model_orig)
    plot_quiver(
        poly_trans,
        poly_const,
        (x_out, y_out),
        "original prediction",
        path_write,
        (xx_trans, yy_trans),
    )

    # repaired model - layer 3
    x_out, y_out = net_meshgrid_prediction(xx, yy, model_repair_3)
    plot_quiver(
        poly_trans,
        poly_const,
        (x_out, y_out),
        "repaired prediction_3",
        path_write,
        (xx_trans, yy_trans),
    )

    # repaired model - layer 2
    x_out, y_out = net_meshgrid_prediction(xx, yy, model_repair_2)
    plot_quiver(
        poly_trans,
        poly_const,
        (x_out, y_out),
        "repaired prediction_2",
        path_write,
        (xx_trans, yy_trans),
    )

    # repaired model - layer 1
    x_out, y_out = net_meshgrid_prediction(xx, yy, model_repair_1)
    plot_quiver(
        poly_trans,
        poly_const,
        (x_out, y_out),
        "repaired prediction_1",
        path_write,
        (xx_trans, yy_trans),
    )


if __name__ == "__main__":
    main()
    pass
