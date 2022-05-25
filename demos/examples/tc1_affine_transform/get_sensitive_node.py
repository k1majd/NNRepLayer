import os
import argparse
from csv import writer
from datetime import datetime
import numpy as np
from affine_utils import (
    plot_dataset,
    model_eval,
    original_data_loader,
    give_polys,
    give_constraints,
)
from shapely.affinity import scale
from tensorflow import keras
from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import (
    ConstraintsClass,
    get_sensitive_nodes,
)
from nnreplayer.repair.repair_weights_class import NNRepair


def arg_parser():
    """_summary_

    Returns:
        _type_: _description_
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        nargs="?",
        default=cwd,
        help="Specify a path to store the data",
    )
    parser.add_argument(
        "-vi",
        "--visualization",
        nargs="?",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Specify visualization variable 1 = on, 0 = off, default: 1",
    )
    parser.add_argument(
        "-sm",
        "--saveModel",
        nargs="?",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Specify whether to save model or not 1 = on, 0 = off, default: 1",
    )
    parser.add_argument(
        "-ss",
        "--saveStats",
        nargs="?",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Specify whether to save stats or not 1 = on, 0 = off, default: 1",
    )
    parser.add_argument(
        "-sms",
        "--saveModelSummery",
        nargs="?",
        type=int,
        default=0,
        choices=range(0, 2),
        help="Specify whether to save model summery or not 1 = on, 0 = off, default: 1",
    )
    parser.add_argument(
        "-rl",
        "--repairLayer",
        nargs="?",
        type=int,
        default=2,
        help="Specify the layer to repair.",
    )
    return parser.parse_args()


def check_log_directories(path_read, path_write, layer_to_repair):
    """_summary_

    Args:
        path_read (_type_): _description_
        path_write (_type_): _description_

    Raises:
        ImportError: _description_
    """
    if not os.path.exists(path_read + "/model"):
        raise ImportError(f"path {path_read}/model does not exist!")

    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print(f"Directory: {path_write} is created!")

    if not os.path.exists(path_write + "/logs"):
        os.makedirs(path_write + "/logs")

    if not os.path.exists(path_write + "/summery"):
        os.makedirs(path_write + "/summery")

    if not os.path.exists(path_write):
        os.makedirs(path_write + f"/model_layer_{layer_to_repair}")

    if not os.path.exists(path_write + "/stats"):
        os.makedirs(path_write + "/stats")


def main(
    direc,
    visual,
    save_model,
    save_stats,
    layer_to_repair,
    save_summery,
):
    """_summary_

    Args:
        direc (_type_): _description_
        visual (_type_): _description_
        save_model (_type_): _description_
        save_stats (_type_): _description_
    """
    print("----------------------")
    print("load model and data")
    # setup directories
    path_read = direc + "/tc1/original_net"
    path_write = direc + "/tc1/repair_net"
    check_log_directories(path_read, path_write, layer_to_repair)

    # load model
    model_orig = keras.models.load_model(path_read + "/model")

    # load dataset and constraints
    x_train, y_train, x_test, y_test = original_data_loader()
    poly_orig, poly_trans, poly_const = give_polys()
    A, b = give_constraints(
        scale(poly_const, xfact=0.98, yfact=0.98, origin="center")
    )

    repair_set = get_sensitive_nodes(
        model_orig, layer_to_repair, x_train, 2, A, b
    )
    print(1)
    # if visual == 1:
    #     print("----------------------")
    #     print("Visualization")
    #     plot_dataset(
    #         [poly_orig, poly_trans, poly_const],
    #         [y_train, out_model.predict(x_train)],
    #         label="training",
    #     )
    #     plot_dataset(
    #         [poly_orig, poly_trans, poly_const],
    #         [y_test, out_model.predict(x_test)],
    #         label="testing",
    #     )


if __name__ == "__main__":
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    args = arg_parser()
    main(
        args.path,
        args.visualization,
        args.saveModel,
        args.saveStats,
        args.repairLayer,
        args.saveModelSummery,
    )
