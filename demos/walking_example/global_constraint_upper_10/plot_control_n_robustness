from cProfile import label
import numpy as np
import os
import csv
import pickle
from csv import writer
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.lines import Line2D
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
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
# plt.rcParams["font.family"] = "Times New Roman"


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
    ax1.set_ylabel("Ankle Angle Control (deg)")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([0, 1000])
    ax1.legend()

    err_orig = np.abs(y_test - pred_ctrls_orig)
    err_repair = np.abs(y_test - pred_ctrls_repair)
    ax2.plot(err_orig, color="#1abd15")
    ax2.plot(err_repair, color="#b81662")
    # ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.set_ylabel("Ankle Angle Control Error (deg)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim([0, 1000])

    ax3.plot(delta_u_orig, color="#1abd15")
    ax3.plot(delta_u_repaired, color="#b81662")
    # ax3.grid(alpha=0.5, linestyle="dashed")
    ax3.axhline(y=bound, color="k", linestyle="dashed")  # upper bound
    ax3.axhline(y=-bound, color="k", linestyle="dashed")  # lower bound
    ax3.set_ylabel("Ankle Angle Control Change (deg)")
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim([0, 1000])

    fig.suptitle(f"Bounded Control, Layer: {layer_to_repair}")
    plt.show()

    # plt.figure(1)
    # plt.plot(test_ctrls, color="#173f5f")
    # plt.plot(pred_ctrls, color=[0.4705, 0.7921, 0.6470])
    # plt.grid(alpha=0.5, linestyle="dashed")
    # # plt.xlim([800,1050])
    # plt.ylabel("Ankle Angle Control (deg)")
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


def plot_mean_vs_std(ax, distance, mean, color, label):
    mean_vec = []
    for i in range(len(distance)):
        mean_vec.append(mean[i]) if mean[i] > 0 else mean_vec.append(0.0)

    # ax.fill_between(
    #     distance,
    #     mean_vec,
    #     lim_uc_vec,
    #     alpha=0.5,
    #     color=color,
    # )
    ax.plot(
        distance,
        mean_vec,
        color=color,
        label=label,
        linewidth=3,
    )
    return ax


if __name__ == "__main__":

    # load finetune and retrain data
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/data/FR_min_dist_global.pickle",
        "rb",
    ) as data:
        dataset2 = pickle.load(data)
    mean_fine = dataset2[0]
    dist_fine = dataset2[1]
    mean_retrain = dataset2[2]
    dist_retrain = dataset2[3]

    # parameter
    bound = 10.0

    load_str3 = "_5_31_2022_16_35_50"
    load_str4 = "_6_9_2022_16_37_31"

    # load test data and original model
    model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/models/model_orig/original_model"
    )
    x_train, y_train, x_test, y_test = generateDataWindow(10)
    y_pred_orig = model_orig.predict(x_test)

    y_temp = model_orig.predict(x_train)
    violate_ids = np.where(y_temp.flatten() > bound)[0]
    x_train = x_train[violate_ids]
    y_train = y_train[violate_ids]

    # load layer 3 data

    model_lay3, dataset_lay3 = generate_model_n_data(load_str3)
    y_pred_lay3 = model_lay3.predict(x_test)

    x_train_lay3 = dataset_lay3[0]
    y_train_lay3 = dataset_lay3[1]

    lim_uc_lay3, mean_lay3, dist_lay3 = give_mean_and_upperstd(
        model_lay3, x_train_lay3, x_train, bound
    )

    # load layer 4 data

    model_lay4, dataset_lay4 = generate_model_n_data(load_str4)
    y_pred_lay4 = model_lay4.predict(x_test)

    x_train_lay4 = dataset_lay4[0]
    y_train_lay4 = dataset_lay4[1]

    lim_uc_lay4, mean_lay4, dist_lay4 = give_mean_and_upperstd(
        model_lay4, x_train_lay4, x_train, bound
    )

    # load original model data
    lim_uc_orig, mean_orig, dist_orig = give_mean_and_upperstd(
        model_orig, x_train_lay4, x_train, bound
    )
    # # find intersection points of y_test and bound
    # y_test_bound = y_test - bound
    # crossing_indices = np.where(np.abs(y_test_bound) <= 0.1)[0]
    # if y_test_bound[crossing_indices[0] - 1] > 0:
    #     crossing_pairs_idx = [(0, crossing_indices[0])]
    #     for i in range(len(crossing_indices) - 2):
    #         crossing_pairs_idx.append(
    #             (crossing_indices[i + 1], crossing_indices[i + 2])
    #         )
    # else:
    #     crossing_pairs_idx = []
    #     for i in range(len(crossing_indices) - 1):
    #         crossing_pairs_idx.append(
    #             (crossing_indices[i], crossing_indices[i + 1])
    #         )
    # create two subplots with share x axis

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig = plt.figure(figsize=(13, 4))
    color_orig = "#2E8B57"
    color_lay3 = "k"
    color_lay4 = "#DC143C"
    color_retrain = "#8E388E"
    color_fine = "#7EC0EE"
    color_test = "black"
    color_xline = "#696969"
    color_fill = "#D4D4D4"
    line_width = 2
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    # share x axis of ax1 and ax2
    ax1.get_shared_x_axes().join(ax1, ax2)
    ax2.get_shared_x_axes().join(ax1, ax2)

    # load original model data
    # fig, (ax1, ax2) = plt.subplots(figsize=(15, 8), nrows=2)
    # fig = plt.figure(figsize=(13, 5))
    # gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    # ax1 = plt.subplot(gs[0])
    xlim = [75, 275]
    ax1.axhline(
        y=bound,
        color=color_xline,
        linewidth=1,
        linestyle=(0, (5, 7)),
        alpha=1,
    )  # upper bound
    ax1.plot(
        y_test[xlim[0] : xlim[1]],
        color=color_test,
        linewidth=line_width,
        linestyle="dashed",
        label="Ref.",
    )
    ax1.plot(
        y_pred_orig[xlim[0] : xlim[1]],
        color=color_orig,
        linewidth=line_width,
        label="Original",
    )
    ax1.plot(
        y_pred_lay3[xlim[0] : xlim[1]],
        color=color_lay3,
        linewidth=line_width,
        label="Repaired - mid layer",
    )
    ax1.plot(
        y_pred_lay4[xlim[0] : xlim[1]],
        color=color_lay4,
        linewidth=line_width,
        label="Repaired - last layer",
    )
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax1.axvspan(23, 47, color=color_fill, alpha=0.5)
    ax1.axvspan(134, 165, color=color_fill, alpha=0.5)
    ax1.axvspan(
        330.7, 355, color=color_fill, alpha=0.5, label="Violated region"
    )
    ax1.set_ylabel("Control [deg]", fontsize=14)
    # ax1.set_xlabel("Time (s)", fontsize=14)
    # ax1.set_xlim([0, 400])
    ax1.set_ylim([-18.0, 21.2])
    # ax1.axes.xaxis.set_visible(False)
    ax1.grid(alpha=0.2, linestyle="dashed", color="#7F7F7F")
    ax2.set_xticks(
        np.linspace(xlim[0] - xlim[0], xlim[1] - xlim[0], 5, endpoint=True)
    )
    ax1.xaxis.set_ticklabels([])
    # ax1.set_xticks(np.linspace(0, 400, 5, endpoint=True))
    ax1.set_yticks(np.linspace(-20, 20, 5, endpoint=True))
    ax1.tick_params(axis="both", which="major", labelsize=14)

    err_orig = np.abs(y_test - y_pred_orig)
    err_lay3 = np.abs(y_test - y_pred_lay3)
    err_lay4 = np.abs(y_test - y_pred_lay4)
    # ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.axvspan(23, 47, color=color_fill, alpha=0.5)
    ax2.axvspan(134, 165, color=color_fill, alpha=0.5)
    # ax2.axvspan(211.3, 239.7, color=color_fill, alpha=0.5)
    # ax2.axvspan(330.7, 355, color=color_fill, alpha=0.5)
    ax2.plot(
        err_orig[xlim[0] : xlim[1]], linewidth=line_width, color=color_orig
    )
    ax2.plot(
        err_lay3[xlim[0] : xlim[1]], linewidth=line_width, color=color_lay3
    )
    ax2.plot(
        err_lay4[xlim[0] : xlim[1]], linewidth=line_width, color=color_lay4
    )
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax2.grid(alpha=0.2, linestyle="dashed", color="#7F7F7F")
    ax2.set_ylabel("Control error [deg]", fontsize=14)
    ax2.set_xlabel("Time [s]", fontsize=14)
    ax2.set_xlim([xlim[0] - xlim[0], xlim[1] - xlim[0]])
    ax2.set_ylim([-0.23, 11.5])
    ax2.set_xticks(
        np.linspace(xlim[0] - xlim[0], xlim[1] - xlim[0], 5, endpoint=True)
    )
    ax2.set_yticks(np.linspace(0, 10, 4, endpoint=True))
    ax2.tick_params(axis="x", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    ax3.set_facecolor("white")
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.grid(alpha=0.2, linestyle="dashed", color="#7F7F7F")
    ax3 = plot_mean_vs_std(
        ax3,
        dist_orig,
        mean_orig(dist_orig),
        # lim_uc_orig,
        color=color_orig,
        label="Original model",
    )
    ax3 = plot_mean_vs_std(
        ax3,
        dist_lay3,
        mean_lay3(dist_lay3),
        # lim_uc_lay3,
        color=color_lay3,
        label="Repaired model - mid layer",
    )
    ax3 = plot_mean_vs_std(
        ax3,
        dist_lay4,
        mean_lay4(dist_lay4),
        # lim_uc_lay4,
        color=color_lay4,
        label="Repaired model - last layer",
    )
    ax3 = plot_mean_vs_std(
        ax3,
        dist_fine,
        mean_fine,
        # lim_uc_fine,
        color=color_fine,
        label="Fine-tuned",
    )
    ax3 = plot_mean_vs_std(
        ax3,
        dist_retrain,
        mean_retrain,
        # lim_uc_retrain,
        color=color_retrain,
        label="Retrained",
    )

    ax3.set_xlabel("$L_2$-distance to the nearest neighbor", fontsize=14)
    ax3.set_ylabel("Violation degree", fontsize=14)
    # ax3.set_title(
    #     "Degree of Violation vs. Distance to Nearest Neighbor", fontsize=14
    # )
    xlim = 2.0
    ylim = 8.8
    ax3.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax3.set_xlim(0, xlim)
    ax3.set_ylim(0 - 0.1, ylim)
    ax3.set_xticks(np.linspace(0, xlim, 5))
    ax3.set_yticks(np.linspace(0, ylim, 3))
    ax3.tick_params(axis="x", labelsize=14)
    ax3.tick_params(axis="y", labelsize=14)

    # ax.set_xlabel("$L_2$-distance to the nearest neighbor", fontsize=14)
    # ax.set_ylabel("Degree of violation", fontsize=14)
    # ax.set_title(
    #     "Degree of Violation vs. Distance to Nearest Neighbor", fontsize=14
    # )
    # ax.set_xlim(0, np.max(dist_orig))
    # ax.set_ylim(0 - 0.1, np.max(lim_uc_orig(dist_orig) + 0.1))
    # ax.set_xticks(np.linspace(0, np.max(dist_orig), 5))
    # ax.set_yticks(np.linspace(0, np.max(lim_uc_orig(dist_orig)), 5))
    # ax.tick_params(axis="x", labelsize=16)
    # ax.tick_params(axis="y", labelsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    lines = lines + lines2[3:]
    labels = labels + labels2[3:]
    leg = fig.legend(
        lines,
        labels,
        loc="center",
        # bbox_to_anchor=(0.5, -0.5),
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        ncol=7,
        fontsize=14,
    )
    leg.get_frame().set_facecolor("white")
    plt.tight_layout()

    plt.show()
    # save data
    print("save data")
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/data/OM_min_dist_dynamic.pickle",
        "wb",
    ) as data:
        pickle.dump(
            [
                mean_orig(dist_orig),
                dist_orig,
                mean_lay3(dist_lay3),
                dist_lay3,
                mean_lay4(dist_lay4),
                dist_lay4,
            ],
            data,
        )
