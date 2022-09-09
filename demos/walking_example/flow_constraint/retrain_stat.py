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
    Dankle = np.subtract(Dtib[:n, 1:], Dfut[:n, 1:])
    observations = np.concatenate(
        (Dfem[:n, 1:], Dtib[:n, 1:], Dankle[:n, 0:1]), axis=1
    )
    observations = (observations - observations.mean(0)) / observations.std(0)
    controls = Dankle[:n, 1]  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    n_train = 18200
    # n_train = 500
    train_observation = np.array([]).reshape(0, 5 * window_size)
    test_observation = np.array([]).reshape(0, 5 * window_size)
    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        train_observation = np.concatenate(
            (train_observation, temp_obs), axis=0
        )
    train_controls = controls[
        window_size - 1 : n_train + window_size - 1
    ].reshape(-1, 1)
    for i in range(n_train, n - window_size):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size - 1 :].reshape(-1, 1)
    return (
        train_observation,
        train_controls,
        test_observation,
        test_controls[:-1],
    )


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


def is_in_box(point, box):
    return box[0] <= point[0] <= box[1] and box[2] <= point[1] <= box[3]


def give_sat_rate(model, x, y, y_pred_prev, box):
    y_pred = model.predict(x)

    num_violate = 0.0
    num_no_violate = 0.0

    for i in range(x.shape[0]):
        point = np.array([x[i, -5] + x[i, -4], x[i, -1] + y_pred_prev[i][0]])
        if is_in_box(point, box):
            num_violate += 1
            point_new = np.array(
                [x[i, -5] + x[i, -4], x[i, -1] + y_pred[i][0]]
            )
            if not is_in_box(point_new, box):
                num_no_violate += 1

    return num_no_violate / num_violate


def give_stats(model, x, y, y_pred_prev, box):
    satisfaction_rate = give_sat_rate(model, x, y, y_pred_prev, box)

    y_pred = model.predict(x)
    y_new = []
    y_orig = []
    for i in range(x.shape[0]):
        point = np.array([x[i, -5] + x[i, -4], x[i, -1] + y_pred_prev[i][0]])
        if not is_in_box(point, box):
            y_new.append(y_pred[i])
            y_orig.append(y[i])
    mae = np.mean(np.abs(np.array(y_orig) - np.array(y_new)))

    # introduced bugs
    idx_orig_no_violate = []
    for i in range(x.shape[0]):
        point = np.array([x[i, -5] + x[i, -4], x[i, -1] + y_pred[i][0]])
        if is_in_box(point, box):
            point2 = np.array(
                [x[i, -5] + x[i, -4], x[i, -1] + y_pred_prev[i][0]]
            )
            if not is_in_box(point2, box):
                idx_orig_no_violate.append(i)

    intro_bug = len(idx_orig_no_violate) / x.shape[0]

    return satisfaction_rate, mae, intro_bug


if __name__ == "__main__":

    x_test, y_test, test_obs, test_ctrls = generateDataWindow(10)

    box = [-2.0, -0.5, 1.0, 3.0]

    for id in range(10):
        model_orig = keras.models.load_model(
            os.path.dirname(os.path.realpath(__file__)) + f"/models/model_orig"
        )

        # load the repaired dat set
        model_repaired = keras.models.load_model(
            os.path.dirname(os.path.realpath(__file__))
            + f"/retrain_model/models/model_{id+1}"
        )

        ctrl_test_pred_orig = model_orig.predict(test_obs)

        sat_rate, mae, intro_bug = give_stats(
            model_repaired, test_obs, test_ctrls, ctrl_test_pred_orig, box
        )

        print(f"mae is {mae}")
        print(f"sat_rate is {sat_rate}")
        print(f"intro_bug is {intro_bug}")

        with open(
            os.path.dirname(os.path.realpath(__file__))
            + f"/retrain_model/stats/bound_stat.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            model_evaluation = [
                "model",
                id,
                "intro_bug",
                intro_bug,
            ]
            # Add contents of list as last row in the csv file
            csv_writer.writerow(model_evaluation)