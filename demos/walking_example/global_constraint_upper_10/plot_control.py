from cProfile import label
import numpy as np
import os
import csv
import pickle
from csv import writer
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D

import tensorflow as tf
from scipy.interpolate import interp1d

# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from shapely.geometry import Polygon
from shapely.affinity import scale

from pyomo.gdp import *
import pyomo.environ as pyo
from scipy.spatial import ConvexHull

from datetime import datetime

from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import ConstraintsClass, get_sensitive_nodes
from nnreplayer.repair.repair_weights_class import NNRepair

plt.rcParams.update({"text.usetex": True})


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
    x_test,
    y_test,
    bound,
    layer_to_repair,
):
    pred_ctrls_orig = original_model.predict(x_test)
    pred_ctrls_repair = repaired_model.predict(x_test)
    delta_u_orig = np.subtract(
        pred_ctrls_orig.flatten(), x_test[:, -1].flatten()
    )
    delta_u_repaired = np.subtract(
        pred_ctrls_repair.flatten(), x_test[:, -1].flatten()
    )
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    ax1.plot(y_test, color="#167fb8", label="Reference")
    ax1.plot(pred_ctrls_orig, color="#1abd15", label="Original Predictions")
    ax1.plot(
        pred_ctrls_repair,
        color="#b81662",
        label="Repaired Predictions",
    )
    ax1.set_ylabel("Ankle Angle Control (rad)")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([0, 1000])
    ax1.legend()

    err_orig = np.abs(y_test - pred_ctrls_orig)
    err_repair = np.abs(y_test - pred_ctrls_repair)
    ax2.plot(err_orig, color="#1abd15")
    ax2.plot(err_repair, color="#b81662")
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.set_ylabel("Ankle Angle Control Error (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim([0, 1000])

    ax3.plot(delta_u_orig, color="#1abd15")
    ax3.plot(delta_u_repaired, color="#b81662")
    ax3.grid(alpha=0.5, linestyle="dashed")
    ax3.axhline(y=bound, color="k", linestyle="dashed")  # upper bound
    ax3.axhline(y=-bound, color="k", linestyle="dashed")  # lower bound
    ax3.set_ylabel("Ankle Angle Control Change (rad)")
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim([0, 1000])

    fig.suptitle(f"Bounded Control, Layer: {layer_to_repair}")
    plt.show()

    # plt.figure(1)
    # plt.plot(test_ctrls, color="#173f5f")
    # plt.plot(pred_ctrls, color=[0.4705, 0.7921, 0.6470])
    # plt.grid(alpha=0.5, linestyle="dashed")
    # # plt.xlim([800,1050])
    # plt.ylabel("Ankle Angle Control (rad)")
    # plt.xlabel("Time (s)")

    # plt.figure(2)

    # print(f"average abs error: {np.sum(err)/err.shape[0]}")
    # direc = os.path.dirname(os.path.realpath(__file__))
    # path_write = os.path.join(direc, "figs")
    # plt.savefig(path_write + f"/repaired_model_32_nodes{now_str}.png")


def generate_repair_dataset(obs, ctrl, num_samples, bound):
    max_window_size = obs.shape[0]
    delta_u = np.subtract(
        ctrl[0:max_window_size].flatten(), obs[0:max_window_size, -1].flatten()
    )
    # violation_idx = np.argsort(np.abs(delta_u))[::-1]
    violation_idx = np.where(np.abs(delta_u) > bound)[0]
    temp = np.argsort(np.abs(delta_u[violation_idx]))[::-1]
    violation_idx = violation_idx[temp]
    nonviolation_idx = np.where(np.abs(delta_u) <= bound)[0]
    if violation_idx.shape[0] == 0:
        nonviolation_idx = np.random.choice(
            nonviolation_idx, size=num_samples, replace=False
        )
        return obs[nonviolation_idx], ctrl[nonviolation_idx]
    else:
        rnd_pts = np.random.choice(
            int(violation_idx.shape[0] / 2), int(num_samples * 0.75)
        )
        violation_idx = violation_idx[rnd_pts]
        nonviolation_idx = np.random.choice(
            nonviolation_idx, size=int(num_samples * 0.25), replace=False
        )
        idx = np.concatenate((violation_idx, nonviolation_idx))
        return obs[idx], ctrl[idx]


def generate_model_n_data(str):
    # load model
    # ctrl_model_orig = keras.models.load_model(
    #     os.path.dirname(os.path.realpath(__file__)) + "/models/model_orig"
    # )

    ctrl_model_repair = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + f"/repair_net/models/model_layer{str}"
    )

    # load data
    if not os.path.exists(
        os.path.dirname(os.path.realpath(__file__)) + "/data"
    ):
        os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/data")
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/data/repair_dataset{str}.pickle",
        "rb",
    ) as data:
        dataset = pickle.load(data)

    return ctrl_model_repair, dataset


def give_mean_and_upperstd(model, x_train, x_test, bound):
    dist = []
    violation = []
    test_pred = model.predict(x_test)
    for i in range(x_test.shape[0]):
        dist.append(np.min(np.linalg.norm(x_train - x_test[i], axis=1)))
        temp = test_pred[i] - bound
        if temp > 0:
            violation.append(temp[0])
        else:
            violation.append(0.0)

    dist = np.array(dist)
    violation = np.array(violation)
    num_bins = 20
    bins = np.linspace(0, np.max(dist), num_bins + 1)
    # mean and std of each bin in violation
    violation_mean = []
    violation_std = []
    for i in range(num_bins):
        idx = np.where(np.logical_and(dist >= bins[i], dist < bins[i + 1]))[0]
        violation_mean.append(np.mean(violation[idx]))
        violation_std.append(np.std(violation[idx]))

    violation_mean = np.array(violation_mean)
    violation_std = np.array(violation_std)

    # plot interpolated violation error bars with mean and fill areas for std
    upper_limit = violation_mean + violation_std
    # lower_limit = violation_mean - violation_std
    lim_uc = interp1d(bins[:-1], upper_limit, kind="cubic")
    # lim_lc = interp1d(bins[:-1], lower_limit, kind="cubic")
    mean = interp1d(bins[:-1], violation_mean, kind="cubic")
    distance = np.linspace(0, np.max(bins[:-1]), 1000)
    return lim_uc, mean, distance


def plot_mean_vs_std(ax, distance, mean, lim_uc, color, label):
    mean_vec = []
    for i in range(len(distance)):
        mean_vec.append(mean(distance[i])) if mean(
            distance[i]
        ) > 0 else mean_vec.append(0.0)
    lim_uc_vec = []
    for i in range(len(distance)):
        lim_uc_vec.append(lim_uc(distance[i])) if lim_uc(
            distance[i]
        ) > mean_vec[i] else lim_uc_vec.append(
            lim_uc(distance[i]) + 2 * (mean_vec[i] - lim_uc(distance[i]))
        )

    ax.fill_between(
        distance,
        mean_vec,
        lim_uc_vec,
        alpha=0.5,
        color=color,
    )
    ax.plot(distance, mean_vec, color=color, label=label)
    return ax


if __name__ == "__main__":
    # parameter
    bound = 10.0

    load_str3 = "_5_31_2022_16_35_50"
    load_str4 = "_6_9_2022_16_37_31"

    # load test data and original model
    model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/models/model_orig/original_model"
    )
    _, _, x_test, y_test = generateDataWindow(10)
    y_pred_orig = model_orig.predict(x_test)

    # load layer 3 data

    model_lay3, _ = generate_model_n_data(load_str3)
    y_pred_lay3 = model_lay3.predict(x_test)

    # load layer 4 data

    model_lay4, _ = generate_model_n_data(load_str4)
    y_pred_lay4 = model_lay4.predict(x_test)

    # load original model data
    fig, (ax1, ax2) = plt.subplots(figsize=(15, 8), nrows=2)
    ax1.axhline(
        y=bound, color="#8B8878", linewidth=3, linestyle="dashed"
    )  # upper bound
    ax1.plot(
        y_test, color="k", linewidth=3, linestyle="dashed", label="Reference"
    )
    ax1.plot(
        y_pred_orig, color="red", linewidth=3, label="Original Predictions"
    )
    ax1.plot(
        y_pred_lay3,
        color="g",
        linewidth=3,
        label="Repaired Predictions - Layer 3",
    )
    ax1.plot(
        y_pred_lay4,
        color="b",
        linewidth=3,
        label="Repaired Predictions - Layer 4",
    )
    ax1.set_ylabel("Ankle Angle Control (rad)", fontsize=16)
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_xlim([0, 400])
    ax1.set_ylim([-18.0, 21.2])
    ax1.grid(alpha=0.8, linestyle="dashed")
    ax1.set_xticks(np.linspace(0, 400, 5, endpoint=True))
    ax1.set_yticks(np.linspace(-20, 20, 5, endpoint=True))
    ax1.tick_params(axis="both", which="major", labelsize=14)

    err_orig = np.abs(y_test - y_pred_orig)
    err_lay3 = np.abs(y_test - y_pred_lay3)
    err_lay4 = np.abs(y_test - y_pred_lay4)
    ax2.plot(err_orig, linewidth=3, color="red")
    ax2.plot(err_lay3, linewidth=3, color="g")
    ax2.plot(err_lay4, linewidth=3, color="b")
    ax2.grid(alpha=0.8, linestyle="dashed")
    ax2.set_ylabel("Ankle Angle Control Error (rad)", fontsize=16)
    ax2.set_xlabel("Time (s)", fontsize=16)
    ax2.set_xlim([0, 400])
    ax2.set_ylim([-0.23, 11.5])
    ax2.set_xticks(np.linspace(0, 400, 5, endpoint=True))
    ax2.set_yticks(np.linspace(0, 10, 4, endpoint=True))
    ax2.tick_params(axis="x", labelsize=16)
    ax2.tick_params(axis="y", labelsize=16)

    # ax.set_xlabel("$L_2$-distance to the nearest neighbor", fontsize=16)
    # ax.set_ylabel("Degree of violation", fontsize=16)
    # ax.set_title(
    #     "Degree of Violation vs. Distance to Nearest Neighbor", fontsize=16
    # )
    # ax.set_xlim(0, np.max(dist_orig))
    # ax.set_ylim(0 - 0.1, np.max(lim_uc_orig(dist_orig) + 0.1))
    # ax.set_xticks(np.linspace(0, np.max(dist_orig), 5))
    # ax.set_yticks(np.linspace(0, np.max(lim_uc_orig(dist_orig)), 5))
    # ax.tick_params(axis="x", labelsize=16)
    # ax.tick_params(axis="y", labelsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(
        lines,
        labels,
        loc="center",
        # bbox_to_anchor=(0.5, -0.5),
        bbox_to_anchor=(0.5, 0.3),
        bbox_transform=fig.transFigure,
        ncol=4,
        fontsize=20,
    )
    leg.get_frame().set_facecolor("white")

    plt.show()
