import os
import numpy as np
from csv import writer
import argparse
from statistics import mode
from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import constraints_class
from nnreplayer.repair.perform_repair import perform_repair
from shapely.geometry import Polygon
from tensorflow import keras
from affine_utils import (
    plot_history,
    plot_dataset,
    model_eval,
    original_data_loader,
    give_polys,
    give_constraints,
)
from affine_utils import give_polys


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
        default=3,
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
        return _squared_sum

    train_dataset = (x_train, y_train)

    weight_slack = 1
    time_limit = 7200
    mip_gap = 0.04
    options = Options(
        "gdp.bigm", "gurobi", "python", "keras", weight_slack, time_limit, mip_gap
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
    print("weight_error: {}".format(results.weight_error))
    print("bias_error: {}".format(results.bias_error))

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

            # Add contents of list as last row in the csv file
            csv_writer.writerow(
                model_eval(
                    results.new_model,
                    keras.models.load_model(path_read + "/model"),
                    path_read,
                    poly_const,
                )
            )
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
