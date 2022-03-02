"""_summary_

Raises:
    ImportError: _description_

Returns:
    _type_: _description_
"""
# pylint: disable=import-error, unused-import
import os
import argparse
import pickle
from statistics import mode
from csv import writer
import numpy as np
from affine_utils import (
    plot_history,
    plot_dataset,
    model_eval,
    original_data_loader,
    give_polys,
    give_constraints,
)
from shapely.geometry import Polygon
from tensorflow import keras
from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import constraints_class
from nnreplayer.repair.perform_repair import perform_repair


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
    # load model
    path_read = direc + "/tc1/original_net"
    path_write = direc + "/tc1/repair_net"
    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print(f"Directory: {path_write} is created!")

    if not os.path.exists(path_read + "/model"):
        raise ImportError(f"path {path_read}/model does not exist!")

    model_orig = keras.models.load_model(path_read + "/model")

    # extract network architecture
    architecture = []
    for lnum, lay in enumerate(model_orig.layers):
        architecture.append(lay.input.shape[1])
        if lnum == len(model_orig.layers) - 1:
            architecture.append(lay.output.shape[1])
    # load dataset and constraints
    x_train, y_train, x_test, y_test = original_data_loader()
    with open(
        path_read + "/data/input_output_data_inside_train_tc1.pickle", "rb"
    ) as data:
        train_in = pickle.load(data)
    with open(
        path_read + "/data/input_output_data_outside_train_tc1.pickle", "rb"
    ) as data:
        train_out = pickle.load(data)
    p = np.random.choice(train_in[0].shape[0], 10)
    x_train_sampled = np.append(train_in[0][p, :], train_out[0], axis=0)
    y_train_sampled = np.append(train_in[1][p, :], train_out[1], axis=0)
    poly_orig, poly_trans, poly_const = give_polys()
    A, b = give_constraints(poly_const)

    print("----------------------")
    print("create repair model")
    # input the constraint list
    constraint_inside = constraints_class("inside", A, b)
    output_constraint_list = [constraint_inside]

    # repair cost
    def squared_sum(x, y):
        m, n = np.array(x).shape
        _squared_sum = 0
        for i in range(m):
            for j in range(n):
                _squared_sum += (x[i, j] - y[i, j]) ** 2
        return _squared_sum / m

    train_dataset = (x_train_sampled, y_train_sampled)

    max_slack = 10.0

    # directory to save optimizer logs
    if not os.path.exists(path_write + "/logs"):
        os.makedirs(path_write + "/logs")

    options = Options(
        "gdp.bigm",
        "gurobi",
        "python",
        "keras",
        max_slack,
        {
            "timelimit": 2500,
            "mipgap": 0.01,
            "mipfocus": 3,
            "improvestarttime": 2000,
        },
        path_write + f"/logs/opt_log_layer{layer_to_repair}.log",
    )
    results = perform_repair(
        layer_to_repair,
        model_orig,
        architecture,
        output_constraint_list,
        squared_sum,
        train_dataset,
        options,
    )

    results.new_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(name="MSE"),
        metrics=["accuracy"],
    )
    results.new_model.evaluate(x_train, y_train, verbose=2)
    print(f"weight_error: {results.weight_error}")
    print(f"bias_error: {results.bias_error}")

    if save_model == 1:
        if not os.path.exists(path_write):
            os.makedirs(path_write + f"/model_{layer_to_repair}")
        keras.models.save_model(
            results.new_model,
            path_write + f"/model_{layer_to_repair}",
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None,
            save_traces=True,
        )
        print("saved: model")

    if save_stats == 1:
        if not os.path.exists(path_write + "/stats"):
            os.makedirs(path_write + "/stats")

        with open(
            path_write + f"/stats/repair_layer{layer_to_repair}_accs_stats_tc1.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            model_evaluation = model_eval(
                results.new_model,
                keras.models.load_model(path_read + "/model"),
                path_read,
                poly_const,
            )
            for key, item in options.optimizer_options.items():
                model_evaluation.append(key)
                model_evaluation.append(item)
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
