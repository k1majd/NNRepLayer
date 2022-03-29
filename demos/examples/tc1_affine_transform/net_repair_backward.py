"""_summary_

Raises:
    ImportError: _description_

Returns:
    _type_: _description_
"""
# pylint: disable=import-error, unused-import
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
        default=3,
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
    path_read_orig_repair = direc + "/tc1/repair_net"
    path_write = direc + "/tc1/repair_net_backward"
    check_log_directories(path_read, path_write, layer_to_repair)

    # load model
    out_model = keras.models.load_model(
        path_read_orig_repair + "/model_layer_3"
    )

    # load dataset and constraints
    x_train, y_train, x_test, y_test = original_data_loader()
    poly_orig, poly_trans, poly_const = give_polys()
    A, b = give_constraints(
        scale(poly_const, xfact=0.98, yfact=0.98, origin="center")
    )

    def mse_weighted_error(data1, data2):
        """return the mean square error of data1-data2 samples

        Args:
            data1 (ndarray): predicted targets
            data2 (ndarray): original targets

        Returns:
            float: mse error
        """
        row, col = np.array(data1).shape
        _squared_sum = 0
        for i in range(row):
            for j in range(col):
                if (np.matmul(A, data2[i, 0:2]) <= b.flatten()).all():
                    _squared_sum += 1 * (data1[i, j] - data2[i, j]) ** 2
                else:
                    _squared_sum += (data1[i, j] - data2[i, j]) ** 2

        return _squared_sum / row

    print("----------------------")
    print("repair model")
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
            "logfile": path_write
            + f"/logs/opt_log_layer{layer_to_repair}.log",
        },
    )

    print(out_model.get_weights())
    for layer_to_repair in [2, 1]:
        print(f"repair layer: {layer_to_repair}")
        repair_obj = NNRepair(out_model)
        repair_obj.compile(
            x_train,
            y_train,
            layer_to_repair,
            output_constraint_list=output_constraint_list,
            cost=mse_weighted_error,
            cost_weights=cost_weights,
            max_weight_bound=max_weight_bound,
        )
        out_model = repair_obj.repair(options)
        print(out_model.get_weights())

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

    print("----------------------")
    print("logging")

    if save_summery == 1:
        repair_obj.summery(direc=path_write + "/summery")
        print("saved: summery")

    if save_model == 1:
        keras.models.save_model(
            out_model,
            path_write + f"/model_backward",
            overwrite=True,
            include_optimizer=False,
            save_format=None,
            signatures=None,
            options=None,
            save_traces=True,
        )
        print("saved: model")

    if save_stats == 1:
        # pylint: disable=unspecified-encoding
        with open(
            path_write + f"/stats/repair_layer_backward_accs_stats_tc1.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            model_evaluation = model_eval(
                out_model,
                keras.models.load_model(path_read + "/model"),
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
        args.saveModelSummery,
    )
