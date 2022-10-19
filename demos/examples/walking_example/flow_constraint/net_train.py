import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorboard
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

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


def plot_pahse(
    ctrl_model_orig,
    obs,
    ctrls,
    var1,
    var2,
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

    fig, ax1 = plt.subplots()
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax1.quiver(
        x,
        y,
        x_vel,
        y_vel,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax1.grid(alpha=0.5, linestyle="dashed")
    ax1.set_xlabel(var1 + " angle")
    ax1.set_ylabel(var2 + " angle")

    ax2.quiver(
        x,
        y,
        x_vel_orig,
        y_vel_orig,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.set_xlabel(var1 + " angle")
    ax2.set_ylabel(var2 + " angle")

    plt.show()


if __name__ == "__main__":
    # Train window model
    (
        train_obs,
        train_ctrls,
        test_obs,
        test_ctrls,
    ) = generateDataWindow(10)
    ctrl_model_orig, callback, architecture = buildModelWindow(train_obs.shape)
    # ctrl_model_orig.load_weights('models/model1')
    ctrl_model_orig.fit(
        train_obs,
        train_ctrls,
        validation_data=(test_obs, test_ctrls),
        batch_size=10,
        epochs=100,
        use_multiprocessing=True,
        verbose=1,
        shuffle=False,
        callbacks=callback,
    )
    keras.models.save_model(
        ctrl_model_orig,
        os.path.dirname(os.path.realpath(__file__)) + "/models/model_orig",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )
    print("saved: model")
    plot_pahse(
        ctrl_model_orig,
        test_obs,
        test_ctrls,
        "femur",
        "ankle",
    )
