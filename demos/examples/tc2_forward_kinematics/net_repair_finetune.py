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
from fk_utils import (
    plot_dataset3d,
    model_eval,
    original_data_loader,
)
import pickle
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
        default=0,
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
    parser.add_argument(
        "-tl",
        "--timeLimit",
        nargs="?",
        type=int,
        default=43200,
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
    time_limit,
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
    path_read = direc + "/tc2/original_net"
    path_repair_orig = direc + "/tc2/repair_net"
    path_write = direc + "/tc2/repair_net_finetune"
    check_log_directories(path_read, path_write, layer_to_repair)

    # load model
    # model_orig = keras.models.load_model(
    #     path_write + f"/model_layer_{layer_to_repair}"
    # )
    model_orig = keras.models.load_model(path_read + "/model")

    # load dataset and constraints
    x_train, y_train, x_test, y_test = original_data_loader()
    with open(
        path_read + "/data/input_output_data_inside_train_tc2.pickle", "rb"
    ) as data:
        train_inside = pickle.load(data)
    rnd_pts = np.random.choice(train_inside[0].shape[0], 100)
    with open(
        path_read + "/data/input_output_data_outside_train_tc2.pickle", "rb"
    ) as data:
        train_out = pickle.load(data)
    x_train = np.vstack((train_inside[0][rnd_pts], train_out[0]))
    y_train = np.vstack((train_inside[1][rnd_pts], train_out[1]))
    A = np.array([[1.0, 0.0, 0.0, 0.0]])
    b = np.array([[0.5]])

    print("----------------------")
    print("repair model")
    # input the constraint list
    constraint_inside = constraints_class("inside", A, b)
    output_constraint_list = [constraint_inside]

    max_weight_bound = 1
    cost_weights = np.array([1.0, 1.0])
    options = Options(
        "gdp.bigm",
        "gurobi",
        "python",
        "keras",
        {
            "timelimit": time_limit,
            "mipgap": 0.001,
            "mipfocus": 1,
            "improvestarttime": time_limit,
            "logfile": path_write
            + f"/logs/opt_log_layer{layer_to_repair}.log",
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
        param_precision=5,
        data_precision=5,
    )
    # initialize the opt model weights with the latest repair weights
    w = keras.models.load_model(
        path_repair_orig + f"/model_layer_{layer_to_repair}"
    ).get_weights()
    for i in range(w[4].shape[0]):
        for j in range(w[4].shape[1]):
            repair_obj.opt_model.w3.set_values({(i, j): w[4][i, j]})
    for i in range(w[5].shape[0]):
        repair_obj.opt_model.b3.set_values({i: w[5][i]})

    out_model = repair_obj.repair(options)

    out_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(name="MSE"),
        metrics=["accuracy"],
    )

    if visual == 1:
        plot_dataset3d(
            [y_train, out_model.predict(x_train)],
            ["hand-labeled output", "fine-tuned output"],
            title_label="training - after repair",
        )
        plot_dataset3d(
            [y_test, out_model.predict(x_test)],
            ["hand-labeled output", "fine-tuned output"],
            title_label="testing - after repair",
        )

    print("----------------------")
    print("logging")

    if save_summery == 1:
        repair_obj.summery(direc=path_write + "/summery")
        print("saved: summery")

    if save_model == 1:
        keras.models.save_model(
            out_model,
            path_write + f"/model_layer_{layer_to_repair}",
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
            path_write
            + f"/stats/repair_layer{layer_to_repair}_accs_stats_tc2.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            model_evaluation = model_eval(
                out_model,
                keras.models.load_model(path_read + "/model"),
                path_read,
                (A, b),
            )
            for key, item in options.optimizer_options.items():
                model_evaluation.append(key)
                model_evaluation.append(item)
            model_evaluation.append("max_weight_bound")
            model_evaluation.append(max_weight_bound)
            model_evaluation.append("cost weights")
            model_evaluation.append(cost_weights)
            model_evaluation.append("repair sample size")
            model_evaluation.append(x_train.shape[0])
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
        args.timeLimit,
    )
