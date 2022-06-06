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
import itertools
from shapely.affinity import scale
from tensorflow import keras
from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import ConstraintsClass, get_sensitive_nodes
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
        default=1,
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
    # repair_set = get_sensitive_nodes(
    #     model_orig, layer_to_repair, x_train, 2, A, b
    # )
    print("----------------------")
    print("repair model")
    # input the constraint list
    constraint_inside = ConstraintsClass("inside", A, b)
    output_constraint_list = [constraint_inside]

    max_weight_bound = 1
    cost_weights = np.array([1.0, 1.0])
    options = Options(
        "gdp.bigm",
        "gurobi",
        "python",
        "keras",
        {
            "timelimit": 360,
            "mipgap": 0.001,
            "mipfocus": 2,
            "improvestarttime": 360,
            "logfile": path_write
            + f"/logs/opt_log_layer__forward_selected.log",
        },
    )

    iter = 1
    for subset in itertools.combinations([0, 1, 2, 3, 4, 5, 6, 7, 8], 5):
        print(f"iteration {iter}")
        iter += 1
        repair_indices = list(subset)
        # repair_indices = [1, 3, 5, 6, 7]
        counter = 0
        col_list = []
        arch = np.array([3, 3, 3, 3])
        for l in range(arch.shape[0] - 1):
            temp = [
                c - counter
                for c in repair_indices
                if (c >= counter and c < counter + arch[l + 1])
            ]
            col_list.append(temp)
            counter += arch[l + 1]
        # print(f)
        # x_train = x_train[0:1, :]
        # y_train = y_train[0:1, :]
        model_orig = keras.models.load_model(path_read + "/model")
        feasib_list = []
        for layer_to_repair, nodes in enumerate(col_list):
            layer_to_repair = layer_to_repair + 1
            if len(nodes) != 0:
                if nodes == [0, 1, 2] and layer_to_repair == 1:
                    model_orig = keras.models.load_model(
                        path_write + "/model_layer_1"
                    )
                    feasib_list.append("f")

                else:
                    repair_obj = NNRepair(model_orig)
                    repair_obj.compile(
                        x_train,
                        y_train,
                        layer_to_repair,
                        output_constraint_list=output_constraint_list,
                        cost_weights=cost_weights,
                        max_weight_bound=max_weight_bound,
                        repair_node_list=nodes,
                        # output_bounds=(-100.0, 100.0),
                    )

                    new_model = repair_obj.repair(options)
                    if repair_obj.opt_model.dw._value is not None:
                        model_orig = new_model
                        feasib_list.append("f")
                    else:
                        feasib_list.append("nf")

        model_orig.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(name="MSE"),
            metrics=["accuracy"],
        )

        # save stats
        with open(
            path_write
            + f"/stats/repair_layer_forward_selected_accs_stats_tc1.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            model_evaluation = model_eval(
                model_orig,
                keras.models.load_model(path_read + "/model"),
                path_read,
                poly_const,
            )
            # for key, item in options.optimizer_options.items():
            #     model_evaluation.append(key)
            #     model_evaluation.append(item)
            model_evaluation.append("selected nodes")
            model_evaluation.append(repair_indices)
            model_evaluation.append("feasibility")
            model_evaluation.append(feasib_list)
            model_evaluation.append("max_weight_bound")
            model_evaluation.append(max_weight_bound)
            model_evaluation.append("cost weights")
            model_evaluation.append(cost_weights)
            model_evaluation.append(str(datetime.now()))
            # Add contents of list as last row in the csv file
            csv_writer.writerow(model_evaluation)
        print("saved: stats")

    if visual == 1:
        print("----------------------")
        print("Visualization")
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

    print("----------------------")
    print("logging")

    # if save_summery == 1:
    #     repair_obj.summery(direc=path_write + "/summery")
    #     print("saved: summery")

    # if save_model == 1:
    #     keras.models.save_model(
    #         model_orig,
    #         path_write + f"/model_layer_forward_selected",
    #         overwrite=True,
    #         include_optimizer=False,
    #         save_format=None,
    #         signatures=None,
    #         options=None,
    #         save_traces=True,
    #     )
    #     print("saved: model")


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