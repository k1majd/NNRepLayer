"""_summary_

Raises:
    ImportError: _description_

Returns:
    _type_: _description_
"""
# pylint: disable=import-error, unused-import
import os
import argparse
from statistics import mode
from csv import writer
from datetime import datetime
import numpy as np
from affine_utils import (
    plot_history,
    plot_dataset,
    model_eval,
    original_data_loader,
    give_polys,
    give_constraints,
)
from shapely.affinity import scale
from shapely.geometry import Polygon, Point
from tensorflow import keras
from matplotlib import pyplot as plt
from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import constraints_class
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
        "-rl",
        "--repairLayer",
        nargs="?",
        type=int,
        default=1,
        help="Specify the layer to repair.",
    )
    return parser.parse_args()


def main(
    direc,
    visual,
    save_model,
    save_stats,
    layer_to_repair,
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
    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print(f"Directory: {path_write} is created!")

    if not os.path.exists(path_read + "/model_2"):
        raise ImportError(f"path {path_read}/model does not exist!")

    if not os.path.exists(path_write + "/logs"):
        os.makedirs(path_write + "/logs")

    # load model
    model_orig = keras.models.load_model(path_read + "/model")

    # load dataset and constraints
    x_train, y_train, x_test, y_test = original_data_loader()
    # p = np.random.choice(x_train.shape[0], 100)
    # x_train_sampled = x_train[p, :]
    # y_train_sampled = y_train[p, :]
    poly_orig, poly_trans, poly_const = give_polys()
    A, b = give_constraints(scale(poly_const, xfact=0.98, yfact=0.98, origin="center"))

    print("----------------------")
    print("create repair model")
    # input the constraint list
    constraint_inside = constraints_class("inside", A, b)
    output_constraint_list = [constraint_inside]

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
            "logfile": path_write + f"/logs/opt_log_layer{layer_to_repair}.log",
        },
    )

    repair_obj = NNRepair(model_orig)
    repair_obj.compile(
        x_train,
        y_train,
        layer_to_repair,
        output_constraint_list=output_constraint_list,
        cost_weights=cost_weights,
        max_weight_bound=max_weight_bound,
    )
    out_model = repair_obj.repair(options)

    out_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(name="MSE"),
        metrics=["accuracy"],
    )

    if visual == 1:
        print("----------------------")
        print("Visualization")
        plot_dataset(
            [poly_orig, poly_trans, poly_const],
            [y_train, out_model.predict(x_train)],
            label="training",
        )
        plot_dataset(
            [poly_orig, poly_trans, poly_const],
            [y_test, out_model.predict(x_test)],
            label="testing",
        )

    if save_model == 1:
        if not os.path.exists(path_write):
            os.makedirs(path_write + f"/model_{layer_to_repair}")
        keras.models.save_model(
            out_model,
            path_write + f"/model_{layer_to_repair}",
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None,
            save_traces=True,
        )
        print(f"saved: model in /{path_write}/model_{layer_to_repair}")

    if save_stats == 1:
        if not os.path.exists(path_write + "/stats"):
            os.makedirs(path_write + "/stats")
        # pylint: disable=unspecified-encoding
        with open(
            path_write + f"/stats/repair_layer{layer_to_repair}_accs_stats_tc1.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            model_evaluation = model_eval(
                out_model,
                keras.models.load_model(path_read + "/model_2"),
                path_read,
                poly_const,
            )
            for key, item in options.optimizer_options.items():
                model_evaluation.append(key)
                model_evaluation.append(item)
            model_evaluation.append("max_weight_bound")
            model_evaluation.append(max_weight_bound)
            model_evaluation.append("cost weights")
            model_evaluation.append(cost_weights)
            model_evaluation.append(str(datetime.now()))
            # Add contents of list as last row in the csv file
            csv_writer.writerow(model_evaluation)
        print("saved: stats")


if __name__ == "__main__":
    args = arg_parser()
    main(
        args.path,
        args.visualization,
        args.saveModel,
        args.saveStats,
        args.repairLayer,
    )
