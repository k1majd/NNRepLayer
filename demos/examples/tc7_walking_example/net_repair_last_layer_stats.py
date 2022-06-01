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


def plotTestData(model, train_obs, train_ctrls, test_obs, test_ctrls):
    pred_ctrls = model(test_obs, training=False)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(test_ctrls, color="#173f5f")
    ax1.plot(pred_ctrls, color=[0.4705, 0.7921, 0.6470])
    ax1.grid(alpha=0.5, linestyle="dashed")
    ax1.set_ylabel("Ankle Angle Control (rad)")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([0, 400])

    err = np.abs(test_ctrls - pred_ctrls)
    ax2.plot(err, color="#173f5f")
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.set_ylabel("Ankle Angle Control Error (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim([0, 400])

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

    # plt.savefig("../figures/layer4_60_n60.pdf")


if __name__ == "__main__":
    # Train window model
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)

    # ctrl_model_orig, callback, architecture = buildModelWindow(train_obs.shape)
    ctrl_model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/models/model_orig/original_model"
    )
    # ctrl_model_orig = keras.models.load_model(
    #     os.path.dirname(os.path.realpath(__file__))
    #     + "/repair_net/models/model_layer_3_5_31_2022_16_35_50"
    # )

    from keras import backend as K

    inp = ctrl_model_orig.input  # input placeholder
    outputs = [
        layer.output for layer in ctrl_model_orig.layers
    ]  # all layer outputs
    functors = [
        K.function([inp], [out]) for out in outputs
    ]  # evaluation functions

    # Testing

    layer_outs = [func([test_obs]) for func in functors]

    # ctrl_model_orig.load_weights('models/model1')
    # ctrl_model_orig.fit(
    #     train_obs,
    #     train_ctrls,
    #     validation_data=(test_obs, test_ctrls),
    #     batch_size=8,
    #     epochs=20,
    #     use_multiprocessing=True,
    #     verbose=1,
    #     shuffle=False,
    #     callbacks=callback,
    # )
    # plotTestData(ctrl_model_orig, train_obs, train_ctrls, test_obs, test_ctrls)

    bound_upper = 10
    bound_lower = 30

    A = np.array([[1], [-1]])
    b = np.array([[bound_upper], [bound_lower]])

    # input the constraint list
    constraint_inside = ConstraintsClass(
        "inside", A, b
    )  # ConstraintsClass(A, b, C, d)
    output_constraint_list = [constraint_inside]
    repair_obj = NNRepair(ctrl_model_orig)

    layer_to_repair = 4  # first layer-(0) last layer-(4)
    max_weight_bound = 10  # specifying the upper bound of weights error
    cost_weights = np.array([100.0, 1.0])  # cost weights
    # output_bounds=np.array([-100.0, 100.0])
    num_samp = [
        1,
        10,
        20,
        50,
        70,
        80,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1500,
        2000,
    ]
    mses = []
    num_exp = 50
    for samp in num_samp:
        temp_mse = 0.0
        for i in range(num_exp):
            rnd_pts = np.random.choice(test_obs.shape[0], samp)
            x_train = test_obs[rnd_pts]
            y_train = test_ctrls[rnd_pts]
            repair_obj.compile(
                x_train,
                y_train,
                layer_to_repair,
                output_constraint_list=output_constraint_list,
                cost_weights=cost_weights,
                max_weight_bound=max_weight_bound,
                output_bounds=(-1e3, 1e3),
            )

            # direc = os.getcwd()
            # path_write = os.path.join(direc, "repair_net")

            # # check directories existence
            # if not os.path.exists(path_write):
            #     os.makedirs(path_write)
            #     print(f"Directory: {path_write} is created!")

            # # setup directory to store optimizer log file
            # if not os.path.exists(path_write + "/logs"):
            #     os.makedirs(path_write + "/logs")

            # # setup directory to store the modeled MIP and parameters
            # if not os.path.exists(path_write + "/summary"):
            #     os.makedirs(path_write + "/summary")

            # # setup directory to store the repaired model
            # if not os.path.exists(path_write):
            #     os.makedirs(path_write + f"/model_layer_{layer_to_repair}")

            # specify options
            options = Options(
                "gdp.bigm",
                "gurobi",
                "python",
                "keras",
                {
                    "timelimit": 3600,  # max time algorithm will take in seconds
                    "mipgap": 0.01,  #
                    "mipfocus": 1,  #
                    "improvestarttime": 3300,
                    # "logfile": path_write
                    # + f"/logs/opt_log_layer{layer_to_repair}.log",
                },
            )

            # repair the network
            out_model = repair_obj.repair(options)

            pred_ctrls = out_model(test_obs, training=False)

            err = np.abs(test_ctrls - pred_ctrls)
            temp_mse += np.sum(err) / err.shape[0]

            repair_obj.reset()
        mses.append(temp_mse / num_exp)

    plt.plot(num_samp, mses)
    plt.xlim([0, 2001])
    plt.ylabel("MSE")
    plt.xlabel("Number of samples")
    plt.title("Output MSE vs Num of Smaples - Last Layer Repair")
    plt.show()

    # store the modeled MIP and parameters
    # repair_obj.summary(direc=path_write + "/summary")

    # store the repaired model
    # keras.models.save_model(
    #     out_model,
    #     path_write + f"/model_layer_{layer_to_repair}",
    #     overwrite=True,
    #     include_optimizer=False,
    #     save_format=None,
    #     signatures=None,
    #     options=None,
    #     save_traces=True,
    # )

    plotTestData(out_model, train_obs, train_ctrls, test_obs, test_ctrls)

    pass
