import numpy as np
import os
import csv
import pickle
from csv import writer
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# import tensorboard
import tensorflow as tf

# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from shapely.geometry import Polygon
from shapely.affinity import scale

from pyomo.gdp import *
from scipy.spatial import ConvexHull

from datetime import datetime

from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import ConstraintsClass
from nnreplayer.repair.repair_weights_class import NNRepair


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    return data


def squared_sum(x, y):
    m, n = np.array(x).shape
    _squared_sum = 0
    for i in range(m):
        for j in range(n):
            _squared_sum += (x[i, j] - y[i, j]) ** 2
    return _squared_sum


def generateDataWindow(window_size):
    Dfem = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_1.csv"
    )
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_2.csv"
    )
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_3.csv"
    )
    n = 20364
    Dankle = np.subtract(Dtib[:n, 1], Dfut[:n, 1])
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    n_train = 18200
    # n_train = 500
    train_observation = np.array([]).reshape(0, 4 * window_size)
    test_observation = np.array([]).reshape(0, 4 * window_size)
    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        train_observation = np.concatenate(
            (train_observation, temp_obs), axis=0
        )
    train_controls = controls[window_size : n_train + window_size].reshape(
        -1, 1
    )
    train_output = observations[window_size : n_train + window_size, :]
    for i in range(n_train, n - window_size):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size :].reshape(-1, 1)
    test_output = observations[n_train + window_size :, :]
    return (
        train_observation,
        np.concatenate((train_controls, train_output), axis=1),
        test_observation,
        np.concatenate((test_controls, test_output), axis=1),
    )


if __name__ == "__main__":
    # record current date n time
    now = datetime.now()
    now_str = f"_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"

    # load data and model
    num_samples = 2
    train_obs, train_out, test_obs, test_out = generateDataWindow(10)
    rnd_pts = np.random.choice(test_obs.shape[0], num_samples)
    x_train = test_obs[rnd_pts]
    y_train = test_out[rnd_pts]

    ctrl_model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__)) + "/models/model_ctrl_pred"
    )

    bound_upper = 10
    bound_lower = 30
    # specify control+predictor architecture
    ctrl_layer_arch = [40, 128, 128, 1]
    pred_layer_arch = [41, 128, 4]
    pred_model_input_order = ["state", "control"]

    A = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]])
    b = np.array([[1.0], [1.5]])

    # input the constraint list
    constraint_inside = ConstraintsClass(
        "inside", A, b
    )  # ConstraintsClass(A, b, C, d)
    output_constraint_list = [constraint_inside]
    repair_obj = NNRepair(ctrl_model_orig)

    layer_to_repair = 4  # first layer-(0) last layer-(4)
    max_weight_bound = 10.0  # specifying the upper bound of weights error
    cost_weights = np.array([1.0, 1.0])  # cost weights
    output_bounds = (-30.0, 40.0)
    repair_node_list = []
    num_nodes = len(repair_node_list) if len(repair_node_list) != 0 else 32
    repair_obj.compile(
        x_train,
        y_train,
        layer_to_repair,
        output_constraint_list=output_constraint_list,
        cost_weights=cost_weights,
        max_weight_bound=max_weight_bound,
        # repair_node_list=repair_set,
        repair_node_list=repair_node_list,
        output_bounds=output_bounds,
    )

    direc = os.path.dirname(os.path.realpath(__file__))
    path_write = os.path.join(direc, "repair_net")

    # check directories existence
    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print(f"Directory: {path_write} is created!")

    # setup directory to store optimizer log file
    if not os.path.exists(path_write + "/logs"):
        os.makedirs(path_write + "/logs")

    # setup directory to store the modeled MIP and parameters
    if not os.path.exists(path_write + "/stats"):
        os.makedirs(path_write + "/stats")

    # # setup directory to store the repaired model
    # if not os.path.exists(path_write):
    #     os.makedirs(
    #         path_write
    #         + f"/models/model_layer_32_nodes_{layer_to_repair}_{num_samples}_{num_nodes}"
    #         + now_str
    #     )

    # specify options
    options = Options(
        "gdp.bigm",
        "gurobi",
        "python",
        "keras",
        {
            "timelimit": 86400,  # max time algorithm will take in seconds
            "mipgap": 0.01,  #
            "mipfocus": 2,  #
            "improvestarttime": 80000,
            "logfile": path_write + f"/logs/opt_log{now_str}.log",
        },
    )

    # repair the network
    out_model = repair_obj.repair(options)

    # store the modeled MIP and parameters
    # repair_obj.summary(direc=path_write + "/summary")

    # store the repaired model
    keras.models.save_model(
        out_model,
        path_write + f"/models/model_layer{now_str}" + now_str,
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )

    if not os.path.exists(
        os.path.dirname(os.path.realpath(__file__)) + "/data"
    ):
        os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/data")
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/data/repair_dataset{now_str}.pickle",
        "wb",
    ) as data:
        pickle.dump([x_train, y_train], data)

    # save summary
    pred_ctrls = out_model(test_obs, training=False)
    err = np.abs(test_ctrls - pred_ctrls)
    with open(
        path_write + f"/stats/repair_layer{now_str}.csv",
        "a+",
        newline="",
    ) as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        model_evaluation = [
            "repair layer",
            layer_to_repair,
            "mae",
            np.sum(err) / err.shape[0],
            "num_samples",
            num_samples,
            "num of repaired nodes",
            num_nodes,
            "repair node list",
            repair_node_list,
            "repair layer",
            layer_to_repair,
            "Num of nodes in repair layer",
            32,
            "timelimit",
            options.optimizer_options["timelimit"],
            "mipfocus",
            options.optimizer_options["mipfocus"],
            "max weight bunds",
            cost_weights,
            "cost weights",
            cost_weights,
            "output bounds",
            output_bounds,
        ]
        # Add contents of list as last row in the csv file
        csv_writer.writerow(model_evaluation)
    print("saved: stats")

    out_model = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/repair_net/models/model_layer_3_5_31_2022_16_35_50"
    )
    # plotTestData(
    #     ctrl_model_orig,
    #     out_model,
    #     train_obs,
    #     train_ctrls,
    #     test_obs,
    #     test_ctrls,
    #     now_str,
    #     bound_upper,
    #     bound_lower,
    #     3,
    # )

    pass
