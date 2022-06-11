import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate

from shapely.geometry import Polygon
from shapely.affinity import scale

from pyomo.gdp import *
from scipy.spatial import ConvexHull

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
    Dfem = loadData("demos/walking_example/data/GeoffFTF_1.csv")
    Dtib = loadData("demos/walking_example/data/GeoffFTF_2.csv")
    Dfut = loadData("demos/walking_example/data/GeoffFTF_3.csv")
    phase = loadData("demos/walking_example/data/GeoffFTF_phase.csv")
    n = 20364
    Dankle = np.subtract(Dtib[:n, 1], Dfut[:n, 1])
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    phase = phase[:n]
    n_train = 19000
    train_observation = np.array([]).reshape(0, 4 * window_size + 1)
    test_observation = np.array([]).reshape(0, 4 * window_size + 1)

    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        temp_obs = np.concatenate(
            (temp_obs, 0 * phase[i + window_size].reshape(1, 1)), axis=1
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
        temp_obs = np.concatenate(
            (temp_obs, 0 * phase[i + window_size].reshape(1, 1)), axis=1
        )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size :].reshape(-1, 1)

    return train_observation, train_controls, test_observation, test_controls


def buildModelWindow(data_size):
    input_size = data_size[0]
    layer_size1 = 32
    layer_size2 = 32
    layer_size3 = 32
    output_size = 1

    model = keras.Sequential(
        [
            layers.Dense(
                layer_size1, activation=tf.nn.relu, input_shape=[input_size]
            ),
            layers.Dense(layer_size2, activation=tf.nn.relu),
            layers.Dense(layer_size3, activation=tf.nn.relu),
            layers.Dense(output_size),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=[tf.keras.losses.MeanAbsoluteError()],
        metrics=["accuracy"],
    )
    model.summary()
    architecture = [
        input_size,
        layer_size1,
        layer_size2,
        layer_size3,
        output_size,
    ]
    filepath = "models/model1"
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
    # plt.xlim([600,900])
    plt.ylabel("Ankle Angle Control (rad)")
    plt.xlabel("Time (s)")
    plt.show()
    # plt.savefig("../figures/layer4_60_n60.pdf")


def plot_pair_vars(var1, var2, dim=1):
    n = 20364
    Dfem = loadData("demos/walking_example/data/GeoffFTF_1.csv")[:n]
    Dtib = loadData("demos/walking_example/data/GeoffFTF_2.csv")[:n]
    Dfut = loadData("demos/walking_example/data/GeoffFTF_3.csv")[:n]
    phase = loadData("demos/walking_example/data/GeoffFTF_phase.csv")[:n]

    Dankle = np.subtract(Dtib[:n], Dfut[:n])

    fig = plt.figure(figsize=(5, 5))

    if var1 == "ankle":
        x = Dankle[:n, dim]
    elif var1 == "femur":
        x = Dfem[:n, dim]
    elif var1 == "shin":
        x = Dtib[:n, dim]
    elif var1 == "foot":
        x = Dfut[:n, dim]
    elif var1 == "phase":
        x = phase

    if var2 == "ankle":
        y = Dankle[:n, dim]
    elif var2 == "femur":
        y = Dfem[:n, dim]
    elif var2 == "shin":
        y = Dtib[:n, dim]
    elif var2 == "foot":
        y = Dfut[:n, dim]
    elif var2 == "phase":
        y = phase

    if dim == 1:
        t_str = " angle"
    elif dim == 2:
        t_str = " velocity"

    plt.plot(x, y, ".")
    plt.xlabel(var1 + t_str)
    plt.ylabel(var2 + t_str)
    plt.grid(alpha=0.5, linestyle="dashed")
    plt.show()


def plot_pahse(var1, var2):
    n = 20364
    Dfem = loadData("demos/walking_example/data/GeoffFTF_1.csv")[:n]
    Dtib = loadData("demos/walking_example/data/GeoffFTF_2.csv")[:n]
    Dfut = loadData("demos/walking_example/data/GeoffFTF_3.csv")[:n]
    phase = loadData("demos/walking_example/data/GeoffFTF_phase.csv")[:n]

    Dankle = np.subtract(Dtib[:n], Dfut[:n])

    if var1 == "ankle":
        x = Dankle[:n]
    elif var1 == "femur":
        x = Dfem[:n]
    elif var1 == "shin":
        x = Dtib[:n]
    elif var1 == "foot":
        x = Dfut[:n]
    elif var1 == "phase":
        x = phase

    if var2 == "ankle":
        y = Dankle[:n]
    elif var2 == "femur":
        y = Dfem[:n]
    elif var2 == "shin":
        y = Dtib[:n]
    elif var2 == "foot":
        y = Dfut[:n]
    elif var2 == "phase":
        y = phase

    fig, ax1 = plt.subplots()

    ax1.quiver(
        x[:, 1],
        y[:, 1],
        x[:, 2],
        y[:, 2],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax1.grid(alpha=0.5, linestyle="dashed")
    ax1.set_xlabel(var1 + " angle")
    ax1.set_ylabel(var2 + " angle")

    plt.show()


if __name__ == "__main__":
    # Train window model
    plot_pair_vars("femur", "ankle", dim=1)
    plot_pahse("femur", "ankle")
