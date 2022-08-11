from threading import BoundedSemaphore
from turtle import color
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
    )  # femur pose?, angle, velocity
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_2.csv"
    )  # shin pose?, angle, velocity
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_3.csv"
    )  # foot? pose?, angle, velocity
    n = 20363
    Dankle = np.subtract(Dtib[: n + 1, 1], Dfut[: n + 1, 1])
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    observations = np.concatenate(
        (
            observations,
            Dankle[:n].reshape(n, 1),
        ),
        axis=1,
    )
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
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


def give_stats(model, x, y, y_pred_prev, bound):
    satisfaction_rate = give_sat_rate(model, x, y, y_pred_prev, bound)
    delta_u_prev = np.subtract(
        y_pred_prev.flatten(), x[:, -1].flatten()
    ).flatten()
    y_pred_new = model.predict(x)

    # find mae
    x_temp = []
    y_orig = []
    y_new = []
    y_prev = []
    for i in range(x.shape[0]):
        if np.abs(delta_u_prev[i]) <= bound:
            x_temp.append(x[i])
            y_orig.append(y[i])
            y_new.append(y_pred_new[i])
            y_prev.append(y_pred_prev[i])
    x_temp = np.array(x_temp)
    y_orig = np.array(y_orig)
    y_new = np.array(y_new)
    mae = np.mean(np.abs(y_orig - y_new))
    mae_prev = np.mean(np.abs(y_orig - y_prev))

    return satisfaction_rate, mae, mae_prev


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
    num_nodes = 128
    model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + f"/models/model_orig_{num_nodes}"
    )

    # load the repaired dat set
    # load_data = "_8_9_2022_13_27_27"
    mae_list = []
    sat_rate_list = []
    l1_norm_list = []
    linf_norm_list = []
    num_rep_weights = []
    mip_gap_list = [
        39.5,
        37.3,
        33.6,
        38.2,
        33.1,
        38.2,
        34.0,
        0,
        37.1,
        33.9,
        32.9,
        37.1,
        33.6,
        2.29,
        36.6,
        37.0,
        37.7,
        36.4,
        37.6,
        39.7,
        41.5,
        37.4,
        0,
        31.4,
        42.0,
        34.0,
        0.0,
        34.2,
        31.3,
        0,
        44.0,
        31.2,
        0,
        29.2,
        0,
    ]
    cost_list = [
        42.99,
        48.15,
        47.33,
        60.18,
        94.56,
        62.62,
        48.32,
        0.0,
        52.13,
        53.63,
        48.21,
        50.24,
        47.98,
        256.01,
        66.03,
        51.26,
        48.27,
        52.10,
        63.85,
        53.88,
        52.02,
        57.71,
        0,
        46.64,
        98.94,
        46.14,
        0.0,
        45.22,
        53.88,
        0,
        62.62,
        70.98,
        0,
        52.45,
        0,
    ]
    num_exp = 35
    for idx in range(num_exp):
        load_str = "_8_10_2022_14_29_12"
        repaired_layer = 2
        model_repaired = keras.models.load_model(
            os.path.dirname(os.path.realpath(__file__))
            + f"/repair_net_relaxed/models/model{idx}_layer{load_str}"
        )
        # x_repair, y_repair = load_rep_data(load_str)

        bound = 2.0

        # original predictions
        ctrl_test_pred_orig = model_orig.predict(test_obs)

        _, mae, mae_prev = give_stats(
            model_repaired, test_obs, test_ctrls, ctrl_test_pred_orig, bound
        )
        sat_rate, _, _ = give_stats(
            model_repaired,
            test_obs,
            test_ctrls,
            ctrl_test_pred_orig,
            bound + 0.2,
        )

        weights_orig = np.concatenate(
            (
                model_orig.get_weights()[2 * (repaired_layer - 1)].flatten(),
                model_orig.get_weights()[
                    2 * (repaired_layer - 1) + 1
                ].flatten(),
            )
        )
        weights_repaired = np.concatenate(
            (
                model_repaired.get_weights()[
                    2 * (repaired_layer - 1)
                ].flatten(),
                model_repaired.get_weights()[
                    2 * (repaired_layer - 1) + 1
                ].flatten(),
            )
        )
        err = weights_orig - weights_repaired
        num_repaired_weights = 0
        for i in range(err.shape[0]):
            if err[i] > 0.0001:
                num_repaired_weights += 1

        # print(f"mae is {mae}")
        # print(f"sat_rate is {sat_rate}")
        # print(f"number of test samples is {test_obs.shape[0]}")
        # print(f"weight l1 norm is {np.linalg.norm(err[err>0.0001], 1)}")
        # print(f"weight l-inf norm is {np.linalg.norm(err[err>0.001], np.inf)}")
        # print(
        #     f"number of repaired weights is {num_repaired_weights}/{err.shape[0]}"
        # )
        if num_repaired_weights == 0:
            mae_list.append(0)
            sat_rate_list.append(0)
            l1_norm_list.append(0)
            linf_norm_list.append(0)
            num_rep_weights.append(0)
        else:
            mae_list.append(mae)
            sat_rate_list.append(sat_rate)
            l1_norm_list.append(np.linalg.norm(err[err > 0.0001], 1))
            linf_norm_list.append(np.linalg.norm(err[err > 0.0001], np.inf))
            num_rep_weights.append(num_repaired_weights)
    # list to np array
    mae_list = np.array(mae_list)
    sat_rate_list = np.array(sat_rate_list)
    l1_norm_list = np.array(l1_norm_list)
    linf_norm_list = np.array(linf_norm_list)
    num_rep_weights = np.array(num_rep_weights)
    mip_gap_list = np.array(mip_gap_list)
    cost_list = np.array(cost_list)

    # bold_idx = np.where(
    #     np.array(mae_list)
    #     == np.min(np.array(mae_list)[np.nonzero(np.array(mae_list))])
    # )[0][0]
    sort_metric = "mae"
    if sort_metric == "mae":
        idx = np.hstack(
            (
                np.delete(
                    np.argsort(mae_list), np.where(np.sort(mae_list) == 0)[0]
                ),
                np.where(mae_list == 0)[0],
            )
        )
    elif sort_metric == "sat_rate":
        idx = np.argsort(sat_rate_list)
        idx = idx[::-1]
    elif sort_metric == "l1_norm":
        idx = np.hstack(
            (
                np.delete(
                    np.argsort(l1_norm_list),
                    np.where(np.sort(l1_norm_list) == 0)[0],
                ),
                np.where(l1_norm_list == 0)[0],
            )
        )
    elif sort_metric == "linf_norm":
        idx = np.hstack(
            (
                np.delete(
                    np.argsort(linf_norm_list),
                    np.where(np.sort(linf_norm_list) == 0)[0],
                ),
                np.where(linf_norm_list == 0)[0],
            )
        )
    elif sort_metric == "num_rep_weights":
        idx = np.hstack(
            (
                np.delete(
                    np.argsort(num_rep_weights),
                    np.where(np.sort(num_rep_weights) == 0)[0],
                ),
                np.where(num_rep_weights == 0)[0],
            )
        )
    elif sort_metric == "mip_gap":
        idx = np.hstack(
            (
                np.delete(
                    np.argsort(mip_gap_list),
                    np.where(np.sort(mip_gap_list) == 0)[0],
                ),
                np.where(mip_gap_list == 0)[0],
            )
        )
        # idx = np.argsort(mip_gap_list)
    elif sort_metric == "cost":
        idx = np.hstack(
            (
                np.delete(
                    np.argsort(cost_list), np.where(np.sort(cost_list) == 0)[0]
                ),
                np.where(cost_list == 0)[0],
            )
        )
        # idx = np.argsort(cost_list)
    # idx = idx[::-1]
    mae_list = mae_list[idx]
    sat_rate_list = sat_rate_list[idx]
    l1_norm_list = l1_norm_list[idx]
    linf_norm_list = linf_norm_list[idx]
    num_rep_weights = num_rep_weights[idx]
    mip_gap_list = mip_gap_list[idx]
    cost_list = cost_list[idx]

    bold_idx = 0

    # print(mae_prev)
    fig = plt.figure(figsize=(10, 10))
    color_orig = "#D4D4D4"
    color_ref = "r"
    color_bold = "#8A8A8A"
    line_width = 2
    width = 0.9

    gs = fig.add_gridspec(3, 2)
    ax00 = fig.add_subplot(gs[0, 0])
    ax10 = fig.add_subplot(gs[1, 0])
    ax20 = fig.add_subplot(gs[2, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax11 = fig.add_subplot(gs[1, 1])
    ax21 = fig.add_subplot(gs[2, 1])
    # ax11 = fig.add_subplot(gs[4, 0])
    # ax01.set_title("Number of repaired weights")
    ax20.set_xlabel("Number of iterations")
    ax21.set_xlabel("Number of iterations")
    ax00.set_ylabel("MAE")
    ax10.set_ylabel("Satisfaction Rate")
    ax20.set_ylabel("L1 norm")
    # ax01.set_ylabel("L-inf norm")
    ax01.set_ylabel("Number of \n repaired weights")
    ax11.set_ylabel("MIP gap")
    ax21.set_ylabel("Opt Cost")
    id = np.where(np.array(mae_list) > 0.001)[0]
    # bold_idx = np.where(
    #     np.array(mae_list)
    #     == np.max(np.array(mae_list)[np.nonzero(np.array(mae_list))])
    # )[0][0]
    # bold_idx = 0
    ax00.bar(
        range(num_exp),
        mae_list,
        color=color_orig,
        # linewidth=line_width,
        width=width,
    )
    ax00.bar(
        bold_idx,
        mae_list[bold_idx],
        color=color_bold,
        width=width,
    )
    # plot max and min line with label
    mae_list = np.array(mae_list)[id]
    ax00.axhline(
        y=np.max(mae_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # upper bound
    ax00.axhline(
        y=np.min(mae_list),
        color="k",
        linewidth=1.5,
        linestyle="dashed",
        label="min/max - partial repair",
    )  # lower bound
    ax00.axhline(
        y=0.5,
        color=color_ref,
        linewidth=1.5,
        linestyle="dashed",
        label="full repair",
    )  # ref
    ax00.axhline(
        y=mae_prev,
        color="b",
        linewidth=1.5,
        linestyle="dashed",
        label="original net",
    )  # ref
    ax00.text(
        1.02,
        np.max(mae_list),
        f"{np.round(np.max(mae_list),2):.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax00.get_yaxis_transform(),
    )
    ax00.text(
        1.02,
        np.min(mae_list),
        f"{np.round(np.min(mae_list),2) :.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax00.get_yaxis_transform(),
    )
    ax00.text(
        0.80,
        0.5,
        f"{np.round(0.5,2) :.2f}",
        va="center",
        ha="left",
        color=color_ref,
        bbox=dict(facecolor="w", alpha=1),
        transform=ax00.get_yaxis_transform(),
    )
    ax00.text(
        0.60,
        mae_prev,
        f"{np.round(mae_prev,2) :.2f}",
        va="center",
        ha="left",
        color="b",
        bbox=dict(facecolor="w", alpha=0.7),
        transform=ax00.get_yaxis_transform(),
    )
    ax00.set_ylim(np.min(mae_list) - 0.05, np.max(mae_list) + 0.05)
    ax00.xaxis.set_ticklabels([])
    ax10.bar(
        range(num_exp),
        sat_rate_list,
        color=color_orig,
        # linewidth=line_width,
        width=width,
    )
    ax10.bar(
        bold_idx,
        sat_rate_list[bold_idx],
        color=color_bold,
        width=width,
    )
    # plot max and min line with label
    sat_rate_list = np.array(sat_rate_list)[id]
    ax10.axhline(
        y=np.max(sat_rate_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # upper bound
    ax10.axhline(
        y=np.min(sat_rate_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # lower bound
    ax10.axhline(
        y=0.93, color=color_ref, linewidth=1.5, linestyle="dashed"
    )  # ref
    ax10.text(
        1.02,
        np.max(sat_rate_list),
        f"{np.max(sat_rate_list):.3f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax10.get_yaxis_transform(),
    )
    ax10.text(
        1.02,
        np.min(sat_rate_list),
        f"{np.min(sat_rate_list) :.3f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax10.get_yaxis_transform(),
    )
    ax10.text(
        0.80,
        0.93,
        f"{np.round(0.93,2) :.2f}",
        va="center",
        ha="left",
        color=color_ref,
        bbox=dict(facecolor="w", alpha=0.7),
        transform=ax10.get_yaxis_transform(),
    )
    ax10.set_ylim(np.min(sat_rate_list) - 0.03, 1)
    ax10.xaxis.set_ticklabels([])
    ax20.bar(
        range(num_exp),
        l1_norm_list,
        color=color_orig,
        # linewidth=line_width,
        width=width,
    )
    ax20.bar(
        bold_idx,
        l1_norm_list[bold_idx],
        color=color_bold,
        width=width,
    )
    l1_norm_list = np.array(l1_norm_list)[id]
    # plot max and min line with its value label in line
    ax20.axhline(
        y=np.max(l1_norm_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # upper bound
    ax20.axhline(
        y=np.min(l1_norm_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # lower bound
    ax20.axhline(
        y=2.42, color=color_ref, linewidth=1.5, linestyle="dashed"
    )  # ref
    ax20.text(
        1.02,
        np.max(l1_norm_list),
        f"{np.round(np.max(l1_norm_list),2) :.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax20.get_yaxis_transform(),
    )
    ax20.text(
        1.02,
        np.min(l1_norm_list),
        f"{np.round(np.min(l1_norm_list),2) :.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax20.get_yaxis_transform(),
    )
    ax20.text(
        0.80,
        2.42,
        f"{np.round(2.42,2) :.2f}",
        va="center",
        ha="left",
        color=color_ref,
        bbox=dict(facecolor="w", alpha=0.7),
        transform=ax20.get_yaxis_transform(),
    )

    def forward(x):
        return x ** (1 / 10)

    def inverse(x):
        return x**10

    ax20.set_yscale("function", functions=(forward, inverse))
    ax20.set_ylim(np.min(l1_norm_list) - 0.5, np.max(l1_norm_list) + 10)
    # ax20.xaxis.set_ticklabels([])
    # ax01.bar(
    #     range(num_exp),
    #     linf_norm_list,
    #     color=color_orig,
    #     linewidth=line_width,
    # )
    ax01.bar(
        range(num_exp),
        num_rep_weights,
        color=color_orig,
        # linewidth=line_width,
        width=width,
    )
    ax01.bar(
        bold_idx,
        num_rep_weights[bold_idx],
        color=color_bold,
        width=width,
    )
    num_rep_weights = np.array(num_rep_weights)[id]
    # plot max and min line with label
    ax01.axhline(
        y=np.max(num_rep_weights), color="k", linewidth=1.5, linestyle="dashed"
    )  # upper bound
    ax01.axhline(
        y=np.min(num_rep_weights), color="k", linewidth=1.5, linestyle="dashed"
    )  # lower bound
    ax01.axhline(
        y=14, color=color_ref, linewidth=1.5, linestyle="dashed"
    )  # ref
    ax01.text(
        1.02,
        np.max(num_rep_weights),
        f"{np.round(np.max(num_rep_weights),2)}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax01.get_yaxis_transform(),
    )
    ax01.text(
        1.02,
        np.min(num_rep_weights),
        f"{np.round(np.min(num_rep_weights),2)}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax01.get_yaxis_transform(),
    )
    ax01.text(
        0.80,
        14,
        f"{np.round(14,2)}",
        va="center",
        ha="left",
        color=color_ref,
        bbox=dict(facecolor="w", alpha=0.7),
        transform=ax01.get_yaxis_transform(),
    )
    ax01.set_ylim(np.min(num_rep_weights) - 2, np.max(num_rep_weights) + 10)

    def forward(x):
        return x ** (1 / 100)

    def inverse(x):
        return x**100

    ax01.set_yscale("function", functions=(forward, inverse))
    ax01.xaxis.set_ticklabels([])
    ax11.bar(
        range(num_exp),
        mip_gap_list,
        color=color_orig,
        # linewidth=line_width,
        width=width,
    )
    ax11.bar(
        bold_idx,
        mip_gap_list[bold_idx],
        color=color_bold,
        width=width,
    )
    mip_gap_list = np.array(mip_gap_list)[id]
    # plot max and min line with label
    ax11.axhline(
        y=np.max(mip_gap_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # upper bound
    ax11.axhline(
        y=np.min(mip_gap_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # lower bound
    ax11.axhline(
        y=38.8, color=color_ref, linewidth=1.5, linestyle="dashed"
    )  # ref
    ax11.text(
        1.02,
        np.max(mip_gap_list),
        f"{np.round(np.max(mip_gap_list),2) :.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax11.get_yaxis_transform(),
    )
    ax11.text(
        1.02,
        np.min(mip_gap_list),
        f"{np.round(np.min(mip_gap_list),2) :.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax11.get_yaxis_transform(),
    )
    ax11.text(
        0.80,
        38.8,
        f"{np.round(38.8,2) :.2f}",
        va="center",
        ha="left",
        color=color_ref,
        bbox=dict(facecolor="w", alpha=0.7),
        transform=ax11.get_yaxis_transform(),
    )
    ax11.set_ylim(np.min(mip_gap_list) - 2, np.max(mip_gap_list) + 2)

    ax11.xaxis.set_ticklabels([])
    ax21.bar(
        range(num_exp),
        cost_list,
        color=color_orig,
        # linewidth=line_width,
        width=width,
    )
    ax21.bar(
        bold_idx,
        cost_list[bold_idx],
        color=color_bold,
        width=width,
    )
    cost_list = np.array(cost_list)[id]
    # plot max and min line with label
    ax21.axhline(
        y=np.max(cost_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # upper bound
    ax21.axhline(
        y=np.min(cost_list), color="k", linewidth=1.5, linestyle="dashed"
    )  # lower bound
    ax21.axhline(
        y=40.29, color=color_ref, linewidth=1.5, linestyle="dashed"
    )  # ref
    ax21.text(
        1.02,
        np.max(cost_list),
        f"{np.round(np.max(cost_list),2) :.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax21.get_yaxis_transform(),
    )
    ax21.text(
        1.02,
        np.min(cost_list),
        f"{np.round(np.min(cost_list),2) :.2f}",
        va="center",
        ha="left",
        color="k",
        bbox=dict(facecolor="w", alpha=0.5),
        transform=ax21.get_yaxis_transform(),
    )
    ax21.text(
        0.80,
        40.29,
        f"{np.round(40.29,2) :.2f}",
        va="center",
        ha="left",
        color=color_ref,
        bbox=dict(facecolor="w", alpha=0.7),
        transform=ax21.get_yaxis_transform(),
    )

    def forward(x):
        return x ** (1 / 500)

    def inverse(x):
        return x**500

    ax21.set_yscale("function", functions=(forward, inverse))
    ax21.set_ylim(np.min(cost_list) - 10, np.max(cost_list) + 10)
    # ax21.xaxis.set_ticklabels([])

    lines, labels = ax00.get_legend_handles_labels()
    leg = fig.legend(
        lines,
        labels,
        loc="center",
        # bbox_to_anchor=(0.5, -0.5),
        bbox_to_anchor=(0.5, 0.01),
        bbox_transform=fig.transFigure,
        ncol=3,
        # fontsize=14,
    )
    leg.get_frame().set_facecolor("white")
    # plt.tight_layout()
    plt.show()

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
