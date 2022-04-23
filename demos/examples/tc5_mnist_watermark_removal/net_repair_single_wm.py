"""_summary_

Raises:
    ImportError: _description_

Returns:
    _type_: _description_
"""
# pylint: disable=import-error, unused-import
import os
import argparse
from csv import DictReader, writer
from datetime import datetime
import pickle
import numpy as np
from wm_utils import (
    # plot_dataset3d,
    # model_eval,
    original_data_loader,
    wm_data_loader,
)
from shapely.affinity import scale
from tensorflow import keras
from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import constraints_class
from nnreplayer.repair.repair_weights_class import NNRepair
import pyomo.environ as pyo
import pyomo.gdp as pyg
from matplotlib import pyplot as plt


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
    path_read = direc + "/tc5/original_net"
    path_write = direc + "/tc5/repair_net_single_wm"
    check_log_directories(path_read, path_write, layer_to_repair)

    # load model
    model_orig = keras.models.load_model(path_read + "/model")

    # load dataset and constraints
    x_train, y_train, x_test, y_test = original_data_loader()
    x_wm, y_wm, label_wm = wm_data_loader(model_orig)
    x_repair = np.array([x_wm[0]])
    y_repair = np.array([y_wm[0]])
    label_repair = np.array([label_wm[0]])
    # with open(
    #     path_read + "/data/input_output_data_inside_train_tc5.pickle", "rb"
    # ) as data:
    #     train_inside = pickle.load(data)
    # rnd_pts = np.random.choice(x_train.shape[0], 200)
    # with open(
    #     path_read + "/data/input_output_data_outside_train_tc5.pickle", "rb"
    # ) as data:
    #     train_out = pickle.load(data)
    # x_train = np.vstack((train_inside[0][rnd_pts], train_out[0]))
    # y_train = np.vstack((train_inside[1][rnd_pts], train_out[1]))
    A = np.array([[1.0, 0.0, 0.0, 0.0]])
    b = np.array([[0.5]])

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
            "mipfocus": 3,
            "improvestarttime": 3300,
            "logfile": path_write
            + f"/logs/opt_log_layer{layer_to_repair}.log",
        },
    )

    # add output disjunctive constraint
    def out_constraint(model, i):
        out_const = []
        max_idx = np.argmax(y_repair[i])
        for k in range(y_repair.shape[1]):
            if k != max_idx:
                out_const.append(
                    [
                        getattr(model, repair_obj.output_name)[i, max_idx]
                        - getattr(model, repair_obj.output_name)[i, k]
                        + 0.0001
                        <= 0
                    ]
                )
        return out_const

    repair_obj = NNRepair(model_orig)
    dw_vec = []
    y_new_vec = []
    for i in range(x_wm.shape[0]):
        x_repair = np.array([x_wm[i]])
        y_repair = np.array([y_wm[i]])
        repair_obj.compile(
            x_repair,
            y_repair,
            layer_to_repair,
            cost_weights=cost_weights,
            max_weight_bound=max_weight_bound,
        )

        setattr(
            repair_obj.opt_model,
            "output_constraint" + str(layer_to_repair),
            pyg.Disjunction(
                range(repair_obj.num_samples), rule=out_constraint
            ),
        )
        # repair_obj.summery(direc=path_write + "/summery")
        out_model = repair_obj.repair(options)
        out_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.MeanSquaredError(name="MSE"),
            metrics=["accuracy"],
        )
        dw_vec.append(repair_obj.opt_model.dw.value)
        y_new_vec.append(out_model.predict(x_repair))
        repair_obj.reset()

    if visual == 1:
        datafile1 = open(direc + "/raw_data/mnist.w.wm.1.wm.csv")
        file_reader = DictReader(datafile1)
        dw_goldberger = np.sort(
            np.array([float(line["sat-epsilon"]) for line in file_reader])
        )
        dw_vec = np.sort(np.array(dw_vec))
        plt.scatter(
            np.linspace(1, 100, 100),
            dw_goldberger,
            label="(goldberger et al. 2020)",
        )
        plt.scatter(
            np.linspace(1, 100, 100),
            dw_vec,
            label="(our method)",
        )
        plt.legend()
        plt.xlabel("watermark images")
        plt.ylabel("delta")
        plt.show()

    print("----------------------")
    print("logging")

    # if save_summery == 1:
    #     repair_obj.summery(direc=path_write + "/summery")
    #     print("saved: summery")

    # if save_model == 1:
    #     keras.models.save_model(
    #         out_model,
    #         path_write + f"/model_layer_{layer_to_repair}",
    #         overwrite=True,
    #         include_optimizer=False,
    #         save_format=None,
    #         signatures=None,
    #         options=None,
    #         save_traces=True,
    #     )
    #     print("saved: model")

    # if save_stats == 1:
    #     # pylint: disable=unspecified-encoding
    #     with open(
    #         path_write
    #         + f"/stats/repair_layer{layer_to_repair}_accs_stats_tc5.csv",
    #         "a+",
    #         newline="",
    #     ) as write_obj:
    #         # Create a writer object from csv module
    #         csv_writer = writer(write_obj)
    #         model_evaluation = model_eval(
    #             out_model,
    #             keras.models.load_model(path_read + "/model"),
    #             path_read,
    #             (A, b),
    #         )
    #         for key, item in options.optimizer_options.items():
    #             model_evaluation.append(key)
    #             model_evaluation.append(item)
    #         model_evaluation.append("max_weight_bound")
    #         model_evaluation.append(max_weight_bound)
    #         model_evaluation.append("cost weights")
    #         model_evaluation.append(cost_weights)
    #         model_evaluation.append(str(datetime.now()))
    #         # Add contents of list as last row in the csv file
    #         csv_writer.writerow(model_evaluation)
    #     print("saved: stats")


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