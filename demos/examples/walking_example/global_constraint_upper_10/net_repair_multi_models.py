import numpy as np
import os
import csv
import pickle
from csv import writer
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from shapely.geometry import Polygon
from shapely.affinity import scale

from pyomo.gdp import *
from scipy.spatial import ConvexHull

from datetime import datetime

from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import ConstraintsClass, get_sensitive_nodes
from nnreplayer.repair.repair_weights_class import NNRepair


def arg_parser():
    """_summary_

    Returns:
        _type_: _description_
    """
    # cwd = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-id",
        "--modelIndex",
        nargs="?",
        type=int,
        default=0,
        help="index of repair model.",
    )
    return parser.parse_args()


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
    for i in range(n_train, n - window_size):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size :].reshape(-1, 1)
    return train_observation, train_controls, test_observation, test_controls


def buildModelWindow(data_size):
    layer_size1 = 32
    layer_size2 = 32
    layer_size3 = 32
    # input_layer = tf.keras.Input(shape=(data_size[1]))
    # layer_1 = layers.Dense(layer_size, activation=tf.nn.relu)(input_layer)
    # layer_2 = layers.Dense(layer_size, activation=tf.nn.relu)(layer_1)
    # layer_3 = layers.Dense(layer_size, activation=tf.nn.relu)(layer_2)
    # output_layer = layers.Dense(1)(layer_3)
    # model = Model(inputs=[input_layer], outputs=[output_layer])
    model = keras.Sequential(
        [
            layers.Dense(
                layer_size1, activation=tf.nn.relu, input_shape=[data_size[1]]
            ),
            layers.Dense(layer_size2, activation=tf.nn.relu),
            layers.Dense(layer_size3, activation=tf.nn.relu),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=[tf.keras.losses.MeanAbsoluteError()],
        metrics=["accuracy"],
    )
    model.summary()
    architecture = [data_size[1], layer_size1, layer_size2, layer_size3, 1]
    filepath = "models/model1"
    # tf_callback=[tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weight_only=False, mode='auto', save_freq='epoch', options=None), keras.callbacks.TensorBoard(log_dir='logs')]
    tf_callback = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weight_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
        ),
        keras.callbacks.TensorBoard(log_dir="tf_logs"),
    ]
    # keras.models.save_model(
    #     model,
    #     os.path.dirname(os.path.realpath(__file__)) + "/original_model",
    #     overwrite=True,
    #     include_optimizer=False,
    #     save_format=None,
    #     signatures=None,
    #     options=None,
    #     save_traces=True,
    # )
    # print("saved: model")
    return model, tf_callback, architecture


def plotTestData(
    original_model,
    repaired_model,
    train_obs,
    train_ctrls,
    test_obs,
    test_ctrls,
    now_str,
    bound_upper,
    bound_lower,
    layer_to_repair,
):
    pred_ctrls_orig = original_model(test_obs, training=False)
    pred_ctrls_repair = repaired_model(test_obs, training=False)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(test_ctrls, color="#167fb8", label="Reference")
    ax1.plot(pred_ctrls_orig, color="#1abd15", label="Original Predictions")
    ax1.plot(
        pred_ctrls_repair,
        color="#b81662",
        label="Repaired Predictions",
    )
    bound_lower = -1 * bound_lower
    ax1.axhline(y=bound_upper, color="k", linestyle="dashed")  # upper bound
    ax1.axhline(y=bound_lower, color="k", linestyle="dashed")  # lower bound
    ax1.set_ylabel("Ankle Angle Control (rad)")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([0, 1000])
    ax1.legend()

    err_orig = np.abs(test_ctrls - pred_ctrls_orig)
    err_repair = np.abs(test_ctrls - pred_ctrls_repair)
    ax2.plot(err_orig, color="#1abd15")
    ax2.plot(err_repair, color="#b81662")
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.set_ylabel("Ankle Angle Control Error (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim([0, 1000])

    # calculate mae

    idx = np.where((test_ctrls > bound_lower) & (bound_upper > test_ctrls))[0]
    fig.suptitle(
        f"Upper-Lower Bound, Layer: {layer_to_repair}, MAE original: {np.mean(err_orig[idx]):.3f}, MAE repaired: {np.mean(err_repair[idx]):.3f}"
    )
    # plt.figure(1)
    # plt.plot(test_ctrls, color="#173f5f")
    # plt.plot(pred_ctrls, color=[0.4705, 0.7921, 0.6470])
    # plt.grid(alpha=0.5, linestyle="dashed")
    # # plt.xlim([800,1050])
    # plt.ylabel("Ankle Angle Control (rad)")
    # plt.xlabel("Time (s)")

    # plt.figure(2)

    print(f"average abs error: {np.sum(err)/err.shape[0]}")
    plt.show()
    # direc = os.path.dirname(os.path.realpath(__file__))
    # path_write = os.path.join(direc, "figs")
    # plt.savefig(path_write + f"/repaired_model_32_nodes{now_str}.png")


def give_stats(model, x, y, y_pred_orig, bound):
    y_pred_new = model.predict(x)

    # find violations
    x_temp = []
    y_temp = []
    for i in range(x.shape[0]):
        if y_pred_orig[i][0] > bound:
            x_temp.append(x[i])
            y_temp.append(y_pred_new[i])

    num_violations = len(x_temp) * 1.0
    num_no_violations = num_violations
    for i in range(len(x_temp)):
        if y_temp[i][0] > bound:
            num_no_violations -= 1.0
    satisfaction_rate = num_no_violations / num_violations

    # find mae
    x_temp = []
    y_orig = []
    y_new = []
    for i in range(x.shape[0]):
        if y_pred_orig[i][0] <= bound:
            x_temp.append(x[i])
            y_orig.append(y[i])
            y_new.append(y_pred_new[i])
    x_temp = np.array(x_temp)
    y_orig = np.array(y_orig)
    y_new = np.array(y_new)
    mae = np.mean(np.abs(y_orig - y_new))

    return satisfaction_rate, mae


if __name__ == "__main__":
    args = arg_parser()
    model_id = args.modelIndex
    # now = datetime.now()
    # now_str = f"_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"
    # Train window model
    num_samples = 150
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)
    idx_adv = np.where(test_ctrls > 10.0)[0]
    id_non_adv = np.where(test_ctrls < 10.0)[0]
    rnd_pts_adv = np.random.choice(
        idx_adv, int(num_samples * 0.75), replace=False
    )
    rnd_pts_non_adv = np.random.choice(
        id_non_adv, int(num_samples * 0.25), replace=False
    )
    rnd_pts = np.concatenate((rnd_pts_adv, rnd_pts_non_adv))
    x_train = test_obs[rnd_pts]
    y_train = test_ctrls[rnd_pts]

    ctrl_model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + f"/models/model_orig_{model_id}"
    )

    bound_upper = 10 - 0.2
    bound_lower = 30

    A = np.array([[1], [-1]])
    b = np.array([[bound_upper], [bound_lower]])

    # input the constraint list
    constraint_inside = ConstraintsClass(
        "inside", A, b
    )  # ConstraintsClass(A, b, C, d)
    output_constraint_list = [constraint_inside]
    repair_obj = NNRepair(ctrl_model_orig)

    layer_to_repair = 3  # first layer-(0) last layer-(4)
    max_weight_bound = 5.0  # specifying the upper bound of weights error
    cost_weights = np.array([1.0, 1.0])  # cost weights
    # output_bounds = (-30.0, 40.0)
    repair_node_list = []
    num_nodes = len(repair_node_list) if len(repair_node_list) != 0 else 32
    repair_obj.compile(
        x_train,
        y_train,
        layer_to_repair,
        output_constraint_list=output_constraint_list,
        cost_weights=cost_weights,
        max_weight_bound=max_weight_bound,
        data_precision=6,
        param_precision=6,
        # repair_node_list=repair_set,
        repair_node_list=repair_node_list,
        w_error_norm=0,
        # output_bounds=output_bounds,
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
            "timelimit": 500,  # max time algorithm will take in seconds
            "mipgap": 0.11,  #
            "mipfocus": 2,  #
            "cuts": 0,
            "concurrentmip": 3,
            "threads": 45,
            "improvestarttime": 400,
            "improvestartgap": 0.12,
            "logfile": path_write + f"/logs/opt_log_multi_{model_id}.log",
        },
    )

    # repair the network
    out_model = repair_obj.repair(options)
    # plt.plot(out_model.predict(x_train))
    # plt.show()

    # store the modeled MIP and parameters
    # repair_obj.summary(direc=path_write + "/summary")

    # store the repaired model
    keras.models.save_model(
        out_model,
        path_write + f"/models/model_layer_multi_{model_id}",
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
        + f"/data/repair_dataset_multi_{model_id}.pickle",
        "wb",
    ) as data:
        pickle.dump([x_train, y_train], data)

    # save summary
    pred_ctrls = out_model(test_obs, training=False)
    err = np.abs(test_ctrls - pred_ctrls)
    ctrl_test_pred_orig = ctrl_model_orig.predict(test_obs)
    _, mae = give_stats(
        out_model, test_obs, test_ctrls, ctrl_test_pred_orig, bound_upper
    )
    sat_rate, _ = give_stats(
        out_model, test_obs, test_ctrls, ctrl_test_pred_orig, bound_upper + 0.2
    )
    with open(
        path_write + f"/stats/repair_layer_multi.csv",
        "a+",
        newline="",
    ) as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        model_evaluation = [
            "model_idx",
            model_id,
            "repair layer",
            layer_to_repair,
            "mae",
            mae,
            "sate_rat",
            sat_rate,
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
            # "output bounds",
            # output_bounds,
        ]
        # Add contents of list as last row in the csv file
        csv_writer.writerow(model_evaluation)
    print("saved: stats")

    # out_model = keras.models.load_model(
    #     os.path.dirname(os.path.realpath(__file__))
    #     + "/repair_net/models/model_layer_3_5_31_2022_16_35_50"
    # )
    # plotTestData(
    #     ctrl_model_orig,
    #     out_model,
    #     train_obs,
    #     train_ctrls,
    #     test_obs,
    #     test_ctrls,
    #     "now",
    #     bound_upper,
    #     bound_lower,
    #     3,
    # )

    # pass
