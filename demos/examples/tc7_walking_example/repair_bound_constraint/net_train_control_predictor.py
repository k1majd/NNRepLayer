import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import tensorboard
import tensorflow as tf

# import tensorflow_probability as tfp
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


def build_network_block(regularizer_rate, layer_size, input, name):
    layer_list = [input]
    for i in range(len(layer_size)):
        activation = tf.nn.relu if i < len(layer_size) - 1 else None
        layer_list.append(
            layers.Dense(
                layer_size[i],
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(regularizer_rate),
                bias_regularizer=keras.regularizers.l2(regularizer_rate),
                name=f"{name}_layer_{i+1}",
            )(layer_list[i])
        )
    return layer_list[-1]


def buildModelWindow(data_size, train_out):
    ctrl_layer_size = [4, 4, 1]
    pred_layer_size = [4, 4]
    regularizer_rate = 0.001

    input_layer = tf.keras.Input(shape=(data_size[1]))
    last_ctrl_layer = build_network_block(
        regularizer_rate, ctrl_layer_size, input_layer, "ctrl"
    )
    last_pred_layer = build_network_block(
        regularizer_rate,
        pred_layer_size,
        layers.Concatenate()([input_layer, last_ctrl_layer]),
        "pred",
    )
    model = Model(
        inputs=[input_layer],
        outputs=[last_pred_layer, last_ctrl_layer],
        name="seq_control_predictor_NN",
    )

    def loss2(y_true, y_pred):
        loss = tf.keras.losses.MSE(y_true, y_pred)
        return loss

    model.compile(
        optimizer="adam",
        loss=[loss2, loss2],
        metrics=["accuracy"],
    )
    model.summary()
    architecture = model.to_json()
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
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]

    return model, tf_callback, architecture


def plotTestData(model, train_obs, train_ctrls, test_obs, test_ctrls):
    pred_ctrls = model.predict(test_obs)

    # subplots
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(test_ctrls[:, 0].flatten(), label="test control", color="#173f5f")
    # axs[0].plot(pred_ctrls[1].flatten(), label="prediction control", color=[0.4705, 0.7921, 0.6470])
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])
    ax5 = fig.add_subplot(gs[:, 1])
    x_lim = 271
    ax5.plot(
        test_ctrls[:, 0].flatten()[:x_lim],
        label="test",
        color="#173f5f",
    )
    ax5.plot(
        pred_ctrls[1].flatten()[:x_lim],
        label="prediction",
        color=[0.4705, 0.7921, 0.6470],
    )
    ax5.set_xlabel("time step")
    ax5.set_ylabel("ankle angle [deg]")
    ax5.set_xlim(0, x_lim)
    # ax5.set_title("Control")
    ax1.plot(
        test_ctrls[:, 1].flatten()[:x_lim],
        label="test femur ang.",
        color="#173f5f",
    )
    ax1.plot(
        pred_ctrls[0][:, 0].flatten()[:x_lim],
        label="pred. femur ang.",
        color=[0.4705, 0.7921, 0.6470],
    )
    ax1.set_ylabel("Femur ang. \n[deg]")
    ax1.set_xlim(0, x_lim)
    # ax1.set_title("Femur Angle")
    ax2.plot(
        test_ctrls[:, 2].flatten()[:x_lim],
        label="test femur ang. vel.",
        color="#173f5f",
    )
    ax2.plot(
        pred_ctrls[0][:, 1].flatten()[:x_lim],
        label="pred. femur ang. vel.",
        color=[0.4705, 0.7921, 0.6470],
    )
    ax2.set_ylabel("Femur ang. vel. \n [deg/s]")
    ax2.set_xlim(0, x_lim)
    # ax2.set_title("Femur Angular Velocity")
    ax3.plot(
        test_ctrls[:, 3].flatten()[:x_lim],
        label="test tibia ang.",
        color="#173f5f",
    )
    ax3.plot(
        pred_ctrls[0][:, 2].flatten()[:x_lim],
        label="pred. tibia ang.",
        color=[0.4705, 0.7921, 0.6470],
    )
    ax3.set_ylabel("Tibia ang. \n [deg]")
    ax3.set_xlim(0, x_lim)
    # ax3.set_title("Tibia Angle")
    ax4.plot(
        test_ctrls[:, 4].flatten()[:x_lim],
        label="test tibia ang. vel.",
        color="#173f5f",
    )
    ax4.plot(
        pred_ctrls[0][:, 3].flatten()[:x_lim],
        label="pred. tibia ang. vel.",
        color=[0.4705, 0.7921, 0.6470],
    )
    ax4.set_xlabel("time step")
    ax4.set_ylabel("Tibia ang. vel. \n [deg/s]")
    ax4.set_xlim(0, x_lim)
    # ax4.set_title("Tibia Angular Velocity")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    # lines, labels = ax5.get_legend_handles_labels()
    # leg = fig.legend(
    #     lines,
    #     labels,
    #     loc="center",
    #     bbox_to_anchor=(0.5, -0.5),
    #     # bbox_to_anchor=(0.75, 0.65),
    #     bbox_transform=fig.transFigure,
    #     ncol=1,
    #     fontsize=14,
    # )
    # leg.get_frame().set_facecolor("white")
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=5120
                    )
                ],
            )
        except RuntimeError as e:
            print(e)
    # Train window model
    (
        train_obs,
        train_out,
        test_obs,
        test_out,
    ) = generateDataWindow(10)
    ctrl_model_orig, callback, architecture = buildModelWindow(
        train_obs.shape, train_out
    )
    # ctrl_model_orig.load_weights('models/model1')
    ctrl_model_orig.fit(
        train_obs,
        [train_out[:, 1:], train_out[:, 0]],
        validation_data=(test_obs, [test_out[:, 1:], test_out[:, 0]]),
        batch_size=20,
        epochs=100,
        use_multiprocessing=True,
        verbose=1,
        shuffle=False,
        callbacks=callback,
    )
    keras.models.save_model(
        ctrl_model_orig,
        os.path.dirname(os.path.realpath(__file__))
        + "/models/model_ctrl_pred_10",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )
    print("saved: model")
    plotTestData(ctrl_model_orig, train_obs, train_out, test_obs, test_out)

    bound_upper = 10
    bound_lower = 30
