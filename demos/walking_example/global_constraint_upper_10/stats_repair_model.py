from threading import BoundedSemaphore
import numpy as np
import os
import csv
import pickle
from csv import writer
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import argparse
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-it",
        "--iteration",
        nargs="?",
        type=int,
        default=1,
        help="iteration of finetune",
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


def load_rep_data(str):
    if not os.path.exists(
        os.path.dirname(os.path.realpath(__file__)) + "/data"
    ):
        os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/data")
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/data/repair_dataset{load_str}.pickle",
        "rb",
    ) as data:
        dataset = pickle.load(data)
    return dataset[0], dataset[1]


def hand_label_samples(x_train, y_train, model, bound, gap=0.1):
    y_pred = model.predict(x_train)
    delta_u_pred = np.subtract(
        y_pred.flatten(), x_train[:, -1].flatten()
    ).flatten()
    x = []
    y = []

    for i in range(x_train.shape[0]):
        if delta_u_pred[i] > 0.7 * bound - gap:
            x.append(x_train[i])
            y.append(np.array([x_train[i, -1] + 0.7 * bound - gap]))
        elif delta_u_pred[i] < -0.7 * bound + gap:
            x.append(x_train[i])
            y.append(np.array([x_train[i, -1] - 0.7 * bound + gap]))
        else:
            x.append(x_train[i])
            y.append(y_train[i])

    return np.array(x), np.array(y)


def give_sat_rate(model, x, y, y_pred_prev, bound):
    delta_u_prev = np.subtract(
        y_pred_prev.flatten(), x[:, -1].flatten()
    ).flatten()
    delta_u_pred = np.subtract(
        model.predict(x).flatten(), x[:, -1].flatten()
    ).flatten()
    delta_u_prev = np.abs(delta_u_prev)
    delta_u_pred = np.abs(delta_u_pred)
    idx = np.where(delta_u_prev > bound)[0]
    new_delta_u_pred = delta_u_pred[idx]
    idx_new = np.where(new_delta_u_pred <= bound)[0]
    return len(idx_new) / len(idx)

    # # find violations
    # x_temp = []
    # y_temp = []
    # for i in range(x.shape[0]):
    #     if y_pred_prev[i][0] > bound:
    #         x_temp.append(x[i])
    #         y_temp.append(y_pred_new[i])

    # num_violations = len(x_temp) * 1.0
    # num_no_violations = num_violations
    # for i in range(len(x_temp)):
    #     if y_temp[i][0] > bound:
    #         num_no_violations -= 1.0
    # satisfaction_rate = num_no_violations / num_violations

    # return satisfaction_rate


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
    iteration = args.iteration

    now = datetime.now()
    now_str = f"_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"
    # Train window model
    # num_samples = 100
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)
    # rnd_pts = np.random.choice(test_obs.shape[0], num_samples)
    # x_train = test_obs[rnd_pts]
    # y_train = test_ctrls[rnd_pts]
    # load the original model
    model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/models/model_orig/original_model"
    )

    # load the repaired dat set
    load_str = "_5_31_2022_16_35_50"
    model_repaired = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + f"/repair_net/models/model_layer{load_str}"
    )
    x_repair, y_repair = load_rep_data(load_str)

    bound = 10.0

    # original predictions
    ctrl_test_pred_orig = model_orig.predict(test_obs)

    sat_rate, mae = give_stats(
        model_repaired, test_obs, test_ctrls, ctrl_test_pred_orig, bound
    )
    print(f"mae is {mae}")
    print(f"sat_rate is {sat_rate}")

    sat_rate, mae = give_stats(
        model_repaired, test_obs, test_ctrls, ctrl_test_pred_orig, bound + 0.1
    )
    print(f"mae is {mae}")
    print(f"sat_rate is {sat_rate}")
    # hand label the data set

    # store the modeled MIP and parameters
    # repair_obj.summary(direc=path_write + "/summary")

    # store the repaired model
    # keras.models.save_model(
    #     out_model,
    #     path_write + f"/models/model_layer{now_str}",
    #     overwrite=True,
    #     include_optimizer=False,
    #     save_format=None,
    #     signatures=None,
    #     options=None,
    #     save_traces=True,
    # )

    # if not os.path.exists(
    #     os.path.dirname(os.path.realpath(__file__)) + "/data"
    # ):
    #     os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/data")
    # with open(
    #     os.path.dirname(os.path.realpath(__file__))
    #     + f"/data/repair_dataset{now_str}.pickle",
    #     "wb",
    # ) as data:
    #     pickle.dump([x_train, y_train], data)

    # save summary

    # # out_model = keras.models.load_model(
    # #     os.path.dirname(os.path.realpath(__file__))
    # #     + "/repair_net/models/model_layer_3_5_31_2022_16_35_50"
    # # )
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

    # pass
