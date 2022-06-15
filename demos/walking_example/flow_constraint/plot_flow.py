from cProfile import label
from turtle import color
import numpy as np
import os
import csv
import pickle
from csv import writer
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pyparsing import line
from matplotlib.ticker import FormatStrFormatter


import tensorflow as tf
from scipy.interpolate import interp1d
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

# import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from shapely.geometry import Polygon
from shapely.affinity import scale
import math
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

    return model, tf_callback, architecture


def plotTestData(model, train_obs, train_ctrls, test_obs, test_ctrls):
    pred_ctrls = model(test_obs, training=False)

    plt.figure(1)
    plt.plot(test_ctrls, color="#173f5f")
    plt.plot(pred_ctrls, color=[0.4705, 0.7921, 0.6470])
    plt.grid(alpha=0.5, linestyle="dashed")
    # plt.xlim([800,1050])
    plt.ylabel("Ankle Angle Control (rad)")
    plt.xlabel("Time (s)")
    plt.show()
    # plt.savefig("../figures/layer4_60_n60.pdf")


def is_in_box(point, box):
    return box[0] <= point[0] <= box[1] and box[2] <= point[1] <= box[3]


def plot_pahse(
    ctrl_model_orig,
    obs,
    ctrls,
    var1,
    var2,
    box,
    ax1,
    c,
    alpha,
    size,
    repair,
    t_max,
    label,
):
    n = 20364
    Dfem = loadData("demos/walking_example/data/GeoffFTF_1.csv")[:n]
    Dtib = loadData("demos/walking_example/data/GeoffFTF_2.csv")[:n]
    Dfut = loadData("demos/walking_example/data/GeoffFTF_3.csv")[:n]
    phase = loadData("demos/walking_example/data/GeoffFTF_phase.csv")[:n]

    Dankle = np.subtract(Dtib[:n], Dfut[:n])

    pred_ankle_vel = ctrl_model_orig.predict(obs)

    if var1 == "ankle":
        x = obs[:, -1].flatten()
        x_vel = pred_ankle_vel.flatten()
        x_vel_orig = ctrls.flatten()
    elif var1 == "femur":
        x = obs[:, -5].flatten()
        x_vel = obs[:, -4].flatten()
        x_vel_orig = obs[:, -4].flatten()
    elif var1 == "shin":
        x = obs[:, -3].flatten()
        x_vel = obs[:, -2].flatten()
        x_vel_orig = obs[:, -2].flatten()

    if var2 == "ankle":
        y = obs[:, -1].flatten()
        y_vel = pred_ankle_vel.flatten()
        y_vel_orig = ctrls.flatten()
    elif var2 == "femur":
        y = obs[:, -5].flatten()
        y_vel = obs[:, -4].flatten()
        y_vel_orig = obs[:, -4].flatten()
    elif var2 == "shin":
        y = obs[:, -3].flatten()
        y_vel = obs[:, -2].flatten()
        y_vel_orig = obs[:, -2].flatten()

    # fig = plt.figure(figsize=(13, 5))
    # gs = fig.add_gridspec(2, 1)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[1, 0])
    # ax1.quiver(
    #     x,
    #     y,
    #     x_vel,
    #     y_vel,
    #     angles="xy",
    #     scale_units="xy",
    #     # scale=1,
    #     color=c,
    #     alpha=alpha,
    # )
    next_x = x + x_vel
    next_y = y + y_vel
    if repair is True:
        for i in range(next_x.shape[0]):
            if is_in_box([next_x[i], next_y[i]], box):
                next_y[i] = box[-2]
    ax1.scatter(
        next_x[0:t_max],
        next_y[0:t_max],
        color=c,
        alpha=alpha,
        s=size,
        label=label,
    )
    # ax1.grid(alpha=0.5, linestyle="dashed")
    # ax1.set_xlabel(var1 + " angle")
    # ax1.set_ylabel(var2 + " angle")
    # plt.xlim([-3.0, 2.6])
    # plt.ylim([-9.5, 4.0])

    # rectangle = plt.Rectangle((0, 0), 20, 20, fc="blue", ec="red")
    # plt.gca().add_patch(rectangle)

    # ax2.quiver(
    #     x,
    #     y,
    #     x_vel_orig,
    #     y_vel_orig,
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    # )
    # ax2.grid(alpha=0.5, linestyle="dashed")
    # ax2.set_xlabel(var1 + " angle")
    # ax2.set_ylabel(var2 + " angle")

    return ax1, next_x, next_y


def plot_signal(x_train, y_train, y_pred, dt=1.0):

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    out_orig = [
        x_train[i, -1] + dt * y_train[i][0] for i in range(x_train.shape[0])
    ]
    out_repaired = [
        x_train[i, -1] + dt * y_pred[i][0] for i in range(x_train.shape[0])
    ]
    ax1.plot(out_orig, label="original")
    ax1.plot(out_repaired, label="repaired")
    ax1.fill_between(
        np.linspace(
            0, x_train[:, -5].shape[0], x_train[:, -5].shape[0], endpoint=True
        ),
        0,
        1,
        where=x_train[:, -5].flatten() < -0.5,
        color="#DDA0DD",
        alpha=0.5,
        transform=ax1.get_xaxis_transform(),
    )
    ax1.axhline(
        y=1, color="#8B8878", linewidth=1.5, linestyle="dashed"
    )  # upper bound
    ax2.plot(x_train[:, -5])
    plt.show()


def is_in_box(point, box):
    return box[0] <= point[0] <= box[1] and box[2] <= point[1] <= box[3]


def generate_repair_dataset(obs, ctrl, num_samples, box, dt=1.0):
    x_train = []
    y_train = []
    for i in range(obs.shape[0]):
        point = np.array([obs[i, -5] + obs[i, -4], obs[i, -1] + ctrl[i][0]])
        if is_in_box(point, box):
            x_train.append(obs[i])
            y_train.append(ctrl[i])
    return np.array(x_train), np.array(y_train)


def give_mean_and_upperstd(
    model_orig, model_rep, x_train, x_test_all, box, bin_size=10
):
    dist = []
    violation_orig = []
    violation_rep = []
    x_test = []
    test_pred_orig_all = model_orig.predict(x_test_all)
    test_pred_rep_all = model_rep.predict(x_test_all)
    for i in range(x_test_all.shape[0]):
        next_x = x_test_all[i, -5] + x_test_all[i, -4]
        next_y_orig = x_test_all[i, -1] + test_pred_orig_all[i][0]
        next_y_rep = x_test_all[i, -1] + test_pred_rep_all[i][0]
        point = np.array([next_x, next_y_orig])
        if is_in_box(point, box):
            x_test.append(x_test_all[i])
            dist.append(
                np.min(np.linalg.norm(x_train - x_test_all[i], axis=1))
            )
            violation_orig.append(next_y_orig - box[-2])
            temp = next_y_rep - box[-2]
            if temp < 0:
                temp = 0
            violation_rep.append(temp)
    dist = np.array(dist)
    violation_orig = np.array(violation_orig)
    violation_rep = np.array(violation_rep)
    num_bins = bin_size
    bins = np.linspace(0, np.max(dist), num_bins + 1)
    # mean and std of each bin in violation
    violation_mean_orig = []
    violation_mean_rep = []
    for i in range(num_bins):
        idx = np.where(np.logical_and(dist >= bins[i], dist < bins[i + 1]))[0]
        violation_mean_orig.append(np.mean(violation_orig[idx]))
        violation_mean_rep.append(np.mean(violation_rep[idx]))

    violation_mean_orig = np.array(violation_mean_orig)
    violation_mean_rep = np.array(violation_mean_rep)

    violate_orig = [
        violation_mean_orig[i]
        for i in range(violation_mean_orig.shape[0])
        if not (math.isnan(violation_mean_orig[i]))
    ]
    violate_rep = [
        violation_mean_rep[i]
        for i in range(violation_mean_rep.shape[0])
        if not (math.isnan(violation_mean_rep[i]))
    ]
    bin_orig = [
        bins[:-1][i]
        for i in range(bins[:-1].shape[0])
        if not (math.isnan(violation_mean_orig[i]))
    ]
    bin_rep = [
        bins[:-1][i]
        for i in range(bins[:-1].shape[0])
        if not (math.isnan(violation_mean_rep[i]))
    ]

    # plot interpolated violation error bars with mean and fill areas for std
    mean_orig = interp1d(bin_orig, violate_orig, kind="cubic")
    mean_rep = interp1d(bin_rep, violate_rep, kind="cubic")
    distance = np.linspace(0, np.max(bins[:-1]), 50)
    return mean_orig, mean_rep, distance


def plot_mean_vs_std(ax, distance, mean, color, line_width):
    mean_vec = []
    for i in range(len(distance)):
        mean_vec.append(mean(distance[i])) if mean(
            distance[i]
        ) > 0 else mean_vec.append(0.0)

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
        linewidth=line_width,
    )
    return ax

    # test_pred_orig = model_orig.predict(x_test)
    # test_pred_rep = model_rep.predict(x_test)
    # for i in range(x_test.shape[0]):
    #     dist.append(np.min(np.linalg.norm(x_train - x_test[i], axis=1)))
    #     temp = test_pred[i] - bound
    #     if temp > 0:
    #         violation.append(temp[0])
    #     else:
    #         violation.append(0.0)

    # dist = np.array(dist)
    # violation = np.array(violation)
    # num_bins = 20
    # bins = np.linspace(0, np.max(dist), num_bins + 1)
    # # mean and std of each bin in violation
    # violation_mean = []
    # violation_std = []
    # for i in range(num_bins):
    #     idx = np.where(np.logical_and(dist >= bins[i], dist < bins[i + 1]))[0]
    #     violation_mean.append(np.mean(violation[idx]))
    #     violation_std.append(np.std(violation[idx]))

    # violation_mean = np.array(violation_mean)
    # violation_std = np.array(violation_std)

    # # plot interpolated violation error bars with mean and fill areas for std
    # upper_limit = violation_mean + violation_std
    # # lower_limit = violation_mean - violation_std
    # lim_uc = interp1d(bins[:-1], upper_limit, kind="cubic")
    # # lim_lc = interp1d(bins[:-1], lower_limit, kind="cubic")
    # mean = interp1d(bins[:-1], violation_mean, kind="cubic")
    # distance = np.linspace(0, np.max(bins[:-1]), 1000)
    # return lim_uc, mean, distance


def cuboid_data(center, size):
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [
        [
            o[0],
            o[0] + l,
            o[0] + l,
            o[0],
            o[0],
        ],  # x coordinate of points in bottom surface
        [
            o[0],
            o[0] + l,
            o[0] + l,
            o[0],
            o[0],
        ],  # x coordinate of points in upper surface
        [
            o[0],
            o[0] + l,
            o[0] + l,
            o[0],
            o[0],
        ],  # x coordinate of points in outside surface
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
    ]  # x coordinate of points in inside surface
    y = [
        [
            o[1],
            o[1],
            o[1] + w,
            o[1] + w,
            o[1],
        ],  # y coordinate of points in bottom surface
        [
            o[1],
            o[1],
            o[1] + w,
            o[1] + w,
            o[1],
        ],  # y coordinate of points in upper surface
        [
            o[1],
            o[1],
            o[1],
            o[1],
            o[1],
        ],  # y coordinate of points in outside surface
        [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
    ]  # y coordinate of points in inside surface
    z = [
        [
            o[2],
            o[2],
            o[2],
            o[2],
            o[2],
        ],  # z coordinate of points in bottom surface
        [
            o[2] + h,
            o[2] + h,
            o[2] + h,
            o[2] + h,
            o[2] + h,
        ],  # z coordinate of points in upper surface
        [
            o[2],
            o[2],
            o[2] + h,
            o[2] + h,
            o[2],
        ],  # z coordinate of points in outside surface
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
    ]  # z coordinate of points in inside surface
    return x, y, z


if __name__ == "__main__":
    str = "_6_11_2022_20_48_45"
    now = datetime.now()
    now_str = f"_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"
    # load model
    box = [-2.0, -0.5, 1.0, 3.0]  # xmin,xmax,ymin,ymax
    ctrl_model_repair = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + f"/repair_net/models/model_layer{str}"
    )
    ctrl_model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__)) + "/models/model_orig"
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
    x_repair = dataset[0]
    y_repair = dataset[1]
    # Train window
    num_samples = 100
    (
        train_obs,
        train_ctrls,
        x_test,
        y_test,
    ) = generateDataWindow(10)
    x_train, y_train = generate_repair_dataset(
        train_obs, train_ctrls, num_samples, box
    )
    # get mean of violations
    mean_orig, mean_rep, distance = give_mean_and_upperstd(
        ctrl_model_orig,
        ctrl_model_repair,
        x_repair,
        train_obs,
        box,
        bin_size=15,
    )
    color_orig = "#DC143C"
    color_lay3 = "k"
    color_box = "#3D3D3D"
    color_xline = "#696969"
    color_fill = "#D4D4D4"
    font_size = 12
    line_width = 1.5

    fig = plt.figure(figsize=(3, 3))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    t_max = train_obs.shape[0]
    ax1, femure_orig, ankle_orig = plot_pahse(
        ctrl_model_orig,
        train_obs,
        train_ctrls,
        "femur",
        "ankle",
        box,
        ax1,
        color_orig,
        alpha=0.7,
        size=1,
        repair=False,
        t_max=t_max,
        label="Original",
    )
    ax1, femure_rep, ankle_rep = plot_pahse(
        ctrl_model_repair,
        train_obs,
        train_ctrls,
        "femur",
        "ankle",
        box,
        ax1,
        color_lay3,
        alpha=0.5,
        size=1,
        repair=True,
        t_max=t_max,
        label="Repaired - mid layer",
    )

    center = [
        (box[1] - box[0]) / 2 + box[0],
        (box[3] - box[2]) / 2 + box[2],
        t_max / 2,
    ]
    length = box[1] - box[0]
    width = box[3] - box[2]
    height = t_max

    # X, Y, Z = cuboid_data(center, (length, width, height))
    # ax1.plot_surface(X, Y, Z, color="b", rstride=1, cstride=1, alpha=0.1)

    if box:
        ax1.plot(
            [box[0], box[0]],
            [box[2], box[3]],
            color=color_box,
            linewidth=line_width,
            linestyle="dashed",
            alpha=1,
        )
        ax1.plot(
            [box[1], box[1]],
            [box[2], box[3]],
            color=color_box,
            linewidth=line_width,
            linestyle="dashed",
            alpha=1,
        )
        ax1.plot(
            [box[0], box[1]],
            [box[2], box[2]],
            color=color_box,
            linewidth=line_width,
            linestyle="dashed",
            alpha=1,
        )
        ax1.plot(
            [box[0], box[1]],
            [box[3], box[3]],
            color=color_box,
            linewidth=line_width,
            linestyle="dashed",
            alpha=1,
        )

    ax1.set_ylabel("Ankle angle (rad)", fontsize=font_size)
    ax1.set_xlabel("Femur angle (rad)", fontsize=font_size)
    ax1.grid(alpha=0.8, linestyle="dashed")
    ax1.tick_params(axis="both", labelsize=font_size)
    ax1.set_xlim([-3.0, 2.6])
    ax1.set_ylim([-9.5, 4.0])
    ax1.set_yticks(np.linspace(-9.5, 4.0, 5, endpoint=True))
    ax1.set_xticks(np.linspace(-3.0, 2.6, 5, endpoint=True))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax1 = fig.add_subplot(gs[0])
    # ax1, femure_orig, ankle_orig = plot_pahse(
    #     ctrl_model_orig,
    #     train_obs,
    #     train_ctrls,
    #     "femur",
    #     "ankle",
    #     box,
    #     ax1,
    #     color_orig,
    #     alpha=1,
    #     size=2,
    #     repair=False,
    # )
    # ax1, femure_rep, ankle_rep = plot_pahse(
    #     ctrl_model_repair,
    #     train_obs,
    #     train_ctrls,
    #     "femur",
    #     "ankle",
    #     box,
    #     ax1,
    #     color_lay3,
    #     alpha=1,
    #     size=2,
    #     repair=True,
    # )
    # # plot box
    # if box:
    #     ax1.plot(
    #         [box[0], box[0]],
    #         [box[2], box[3]],
    #         color=color_box,
    #         linewidth=line_width,
    #         linestyle="dashed",
    #         alpha=0.5,
    #     )
    #     ax1.plot(
    #         [box[1], box[1]],
    #         [box[2], box[3]],
    #         color=color_box,
    #         linewidth=line_width,
    #         linestyle="dashed",
    #         alpha=0.5,
    #     )
    #     ax1.plot(
    #         [box[0], box[1]],
    #         [box[2], box[2]],
    #         color=color_box,
    #         linewidth=line_width,
    #         linestyle="dashed",
    #         alpha=0.5,
    #     )
    #     ax1.plot(
    #         [box[0], box[1]],
    #         [box[3], box[3]],
    #         color=color_box,
    #         linewidth=line_width,
    #         linestyle="dashed",
    #         alpha=0.5,
    #     )

    # ax1.set_ylabel("Ankle angle (rad)", fontsize=font_size)
    # ax1.set_xlabel("Femur angle (rad)", fontsize=font_size)
    # ax1.grid(alpha=0.8, linestyle="dashed")
    # ax1.tick_params(axis="both", labelsize=font_size)
    # ax1.set_xlim([-3.0, 2.6])
    # ax1.set_ylim([-9.5, 4.0])
    # ax1.set_yticks(np.linspace(-9.5, 4.0, 5, endpoint=True))
    # ax1.set_xticks(np.linspace(-3.0, 2.6, 5, endpoint=True))
    # ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    # ax1.rcParams.update({"text.usetex": True})

    lines, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(
        lines,
        labels,
        loc="center",
        # bbox_to_anchor=(0.5, -0.5),
        bbox_to_anchor=(0.3, 0.1),
        bbox_transform=fig.transFigure,
        ncol=1,
        fontsize=font_size,
        frameon=False,
    )
    leg.get_frame().set_facecolor("white")
    # leg.get_frame().set_framealpha(0.0)
    # plt.legend(frameon=False)
    plt.tight_layout()

    plt.show()
    # plot_signal(
    #     train_obs,
    #     train_ctrls,
    #     ctrl_model_repair.predict(train_obs),
    #     dt=1.,
    # )
    # plotTestData(
    #     ctrl_model_orig,
    #     out_model,
    #     x_test,
    #     y_test,
    #     now_str,
    #     bound,
    #     layer_to_repair,
    # )
