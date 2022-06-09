import numpy as np
import os
import csv
import pickle
from csv import writer
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


import tensorflow as tf

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

from scipy.interpolate import interp1d


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
            (temp_obs, phase[i + window_size - 1].reshape(1, 1)), axis=1
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
            (temp_obs, phase[i + window_size - 1].reshape(1, 1)), axis=1
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
    repair_obs,
    repair_ctrls,
    now_str,
    layer_to_repair,
):

    lim_x = np.array(
        [
            0.00,
            0.075,
            0.12,
            0.30,
            0.40,
            0.46,
            0.485,
            0.56,
            0.625,
            0.675,
            0.78,
            0.885,
            0.98,
            1,
        ]
    )
    lim_u = (
        np.array(
            [9, 3.0, 4.0, 13, 18, 22.0, 23.0, 15.0, 0.0, 1.0, 9.0, 4.5, 11, 9]
        )
        + 1
    )
    lim_l = (
        np.array(
            [0, -5, -4.0, 6.0, 11, 14.0, 15, -2.0, -14, -11, -1.0, -6.5, 1, 0]
        )
        - 0.8
    )

    new_x = np.linspace(0, 1, 51)
    lim_uc = interp1d(lim_x, lim_u, kind="cubic")
    lim_lc = interp1d(lim_x, lim_l, kind="cubic")

    lim_u = np.interp(new_x, lim_x, lim_u)
    lim_l = np.interp(new_x, lim_x, lim_l)

    pred_ctrls_repair_test = repaired_model.predict(test_obs)
    pred_ctrls_orig_test = original_model.predict(test_obs)
    pred_ctrls_repair_train = repaired_model.predict(train_obs)
    pred_ctrls_orig_train = original_model.predict(train_obs)
    pred_ctrls_repair_repair = repaired_model.predict(repair_obs)
    pred_ctrls_orig_repair = original_model.predict(repair_obs)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    ax1.scatter(test_obs[:, -1], pred_ctrls_orig_test, label="original - test")
    ax1.scatter(
        test_obs[:, -1], pred_ctrls_repair_test, label="repaired - test"
    )

    ax2.scatter(
        train_obs[:, -1], pred_ctrls_orig_train, label="original - train"
    )
    ax2.scatter(
        train_obs[:, -1], pred_ctrls_repair_train, label="repaired - train"
    )

    ax3.scatter(
        repair_obs[:, -1], pred_ctrls_orig_repair, label="original - repair"
    )
    ax3.scatter(
        repair_obs[:, -1], pred_ctrls_repair_repair, label="repaired - repair"
    )

    ax1.plot(new_x, lim_uc(new_x), "--r", label="upper limit")
    ax1.plot(new_x, lim_lc(new_x), "--r", label="lower limit")

    ax2.plot(new_x, lim_uc(new_x), "--r", label="upper limit")
    ax2.plot(new_x, lim_lc(new_x), "--r", label="lower limit")

    ax3.plot(new_x, lim_uc(new_x), "--r", label="upper limit")
    ax3.plot(new_x, lim_lc(new_x), "--r", label="lower limit")

    ax1.legend()
    ax2.legend()
    ax3.legend()

    ax1.grid()
    ax2.grid()
    ax3.grid()
    # plt.plot(new_x, lim_u, 'k')
    # plt.plot(new_x, lim_l, 'k')
    plt.show()

    # # pred_ctrls_orig = original_model.predict(test_obs)
    # pred_ctrls_repair = repaired_model.predict(test_obs)
    # delta_u_orig = np.subtract(
    #     pred_ctrls_orig.flatten(), test_obs[:, -1].flatten()
    # )
    # delta_u_repaired = np.subtract(
    #     pred_ctrls_repair.flatten(), test_obs[:, -1].flatten()
    # )
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    # ax1.plot(test_ctrls, color="#167fb8", label="Reference")
    # ax1.plot(pred_ctrls_orig, color="#1abd15", label="Original Predictions")
    # ax1.plot(
    #     pred_ctrls_repair,
    #     color="#b81662",
    #     label="Repaired Predictions",
    # )
    # ax1.set_ylabel("Ankle Angle Control (rad)")
    # ax1.set_xlabel("Time (s)")
    # ax1.set_xlim([0, 1000])
    # ax1.legend()

    # err_orig = np.abs(test_ctrls - pred_ctrls_orig)
    # err_repair = np.abs(test_ctrls - pred_ctrls_repair)
    # ax2.plot(err_orig, color="#1abd15")
    # ax2.plot(err_repair, color="#b81662")
    # ax2.grid(alpha=0.5, linestyle="dashed")
    # ax2.set_ylabel("Ankle Angle Control Error (rad)")
    # ax2.set_xlabel("Time (s)")
    # ax2.set_xlim([0, 1000])

    # ax3.plot(delta_u_orig, color="#1abd15")
    # ax3.plot(delta_u_repaired, color="#b81662")
    # ax3.grid(alpha=0.5, linestyle="dashed")
    # ax3.axhline(y=2, color="k", linestyle="dashed")  # upper bound
    # ax3.axhline(y=-2, color="k", linestyle="dashed")  # lower bound
    # ax3.set_ylabel("Ankle Angle Control Change (rad)")
    # ax3.set_xlabel("Time (s)")
    # ax3.set_xlim([0, 1000])

    # fig.suptitle(f"Bounded Control, Layer: {layer_to_repair}")
    # plt.show()

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


def generate_repair_dataset(obs, ctrl, num_samples, num_bins=10):
    x_train = np.array([]).reshape(0, obs.shape[1])
    y_train = np.array([]).reshape(0, ctrl.shape[1])
    phase_vs_limits = loadData(
        "demos/walking_example/data/GeoffFTF_limits.csv"
    )
    adv_id = []
    nonadv_id = []
    for i in range(ctrl.shape[0]):
        diff = np.abs(phase_vs_limits[:, 0] - obs[i, -1])
        idx = diff.argmin()
        if not (
            (ctrl[i] <= phase_vs_limits[idx, 1])
            and (ctrl[i] >= phase_vs_limits[idx, 2])
        ):
            adv_id.append(i)
            # x_train.append(obs[i])
            # y_train.append(ctrl[i])
        else:
            nonadv_id.append(i)

    num_adv_per_bin = int(0.75 * num_samples / num_bins)
    num_nonadv_per_bin = int(0.25 * num_samples / num_bins)

    adv_obs = obs[adv_id, :]
    adv_ctrl = ctrl[adv_id]

    nonadv_obs = obs[nonadv_id, :]
    nonadv_ctrl = ctrl[nonadv_id]

    bin = np.linspace(0, 1, num_bins + 1)
    for i in range(num_bins):
        adv_idx = np.random.choice(
            np.where(
                (adv_obs[:, -1] > bin[i]) & (adv_obs[:, -1] < bin[i + 1])
            )[0],
            num_adv_per_bin,
            replace=False,
        )

        nonadv_idx = np.random.choice(
            np.where(
                (nonadv_obs[:, -1] > bin[i]) & (nonadv_obs[:, -1] < bin[i + 1])
            )[0],
            num_nonadv_per_bin,
            replace=False,
        )
        x_train = np.concatenate((x_train, adv_obs[adv_idx, :]), axis=0)
        y_train = np.concatenate((y_train, adv_ctrl[adv_idx]), axis=0)
        x_train = np.concatenate((x_train, nonadv_obs[nonadv_idx, :]), axis=0)
        y_train = np.concatenate((y_train, nonadv_ctrl[nonadv_idx]), axis=0)

    return x_train, y_train


if __name__ == "__main__":
    now = datetime.now()
    now_str = f"_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"
    # Train window model
    phase_vs_limits = loadData(
        "demos/walking_example/data/GeoffFTF_limits.csv"
    )
    num_samples = 100
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)
    x_train, y_train = generate_repair_dataset(
        train_obs, train_ctrls, num_samples
    )

    ctrl_model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__)) + "/models/model_orig"
    )
    # plotTestData(
    #     ctrl_model_orig,
    #     train_obs,
    #     train_ctrls,
    #     test_obs,
    #     test_ctrls,
    #     now_str,
    # )

    def out_constraint_upper(model, i):
        diff = np.abs(phase_vs_limits[:, 0] - x_train[i, -1])
        idx = diff.argmin()
        return (
            getattr(model, repair_obj.output_name)[i, 0]
            <= phase_vs_limits[idx, 1]
        )

    def out_constraint_lower(model, i):
        diff = np.abs(phase_vs_limits[:, 0] - x_train[i, -1])
        idx = diff.argmin()
        return (
            getattr(model, repair_obj.output_name)[i, 0]
            >= phase_vs_limits[idx, 2]
        )

    repair_obj = NNRepair(ctrl_model_orig)

    layer_to_repair = 4  # first layer-(0) last layer-(4)
    max_weight_bound = 10  # specifying the upper bound of weights error
    cost_weights = np.array([10.0, 1.0])  # cost weights
    output_bounds = (-30.0, 100.0)
    repair_node_list = []
    num_nodes = len(repair_node_list) if len(repair_node_list) != 0 else 32
    repair_obj.compile(
        x_train,
        y_train,
        layer_to_repair,
        # output_constraint_list=output_constraint_list,
        cost_weights=cost_weights,
        max_weight_bound=max_weight_bound,
        data_precision=4,
        param_precision=4,
        # repair_node_list=repair_set,
        repair_node_list=repair_node_list,
        output_bounds=output_bounds,
    )
    setattr(
        repair_obj.opt_model,
        "out_constraint_upper" + str(layer_to_repair),
        pyo.Constraint(
            range(repair_obj.num_samples), rule=out_constraint_upper
        ),
    )
    setattr(
        repair_obj.opt_model,
        "out_constraint_lower" + str(layer_to_repair),
        pyo.Constraint(
            range(repair_obj.num_samples), rule=out_constraint_lower
        ),
    )

    direc = os.path.dirname(os.path.realpath(__file__))
    path_write = os.path.join(direc, "repair_net")

    # check directories existence
    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print(f"Directory: {path_write} is created!")

    # setup directory to store optimizer log file
    if not os.path.exists(path_write + "/logs"):
        os.makedirs(path_write + "/logs")

    # setup directory to store the modeled MIP and parameters
    if not os.path.exists(path_write + "/stats"):
        os.makedirs(path_write + "/stats")

    # # setup directory to store the repaired model
    # if not os.path.exists(path_write):
    #     os.makedirs(
    #         path_write
    #         + f"/models/model_layer_32_nodes_{layer_to_repair}_{num_samples}_{num_nodes}"
    #         + now_str
    #     )

    # specify options
    options = Options(
        "gdp.bigm",
        "gurobi",
        "python",
        "keras",
        {
            "timelimit": 3600,  # max time algorithm will take in seconds
            "mipgap": 0.01,  #
            "mipfocus": 2,  #
            "improvestarttime": 3000,
            "logfile": path_write + f"/logs/opt_log{now_str}.log",
        },
    )

    # repair the network
    out_model = repair_obj.repair(options)

    # store the modeled MIP and parameters
    # repair_obj.summary(direc=path_write + "/summary")

    # store the repaired model
    keras.models.save_model(
        out_model,
        path_write + f"/models/model_layer{now_str}" + now_str,
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )

    if not os.path.exists(
        os.path.dirname(os.path.realpath(__file__)) + "/data"
    ):
        os.makedirs(os.path.dirname(os.path.realpath(__file__)) + "/data")
    with open(
        os.path.dirname(os.path.realpath(__file__))
        + f"/data/repair_dataset{now_str}.pickle",
        "wb",
    ) as data:
        pickle.dump([x_train, y_train], data)

    # save summary
    pred_ctrls = out_model(test_obs, training=False)
    err = np.abs(test_ctrls - pred_ctrls)
    with open(
        path_write + f"/stats/repair_layer{now_str}.csv",
        "a+",
        newline="",
    ) as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        model_evaluation = [
            "repair layer",
            layer_to_repair,
            "mae",
            np.sum(err) / err.shape[0],
            "num_samples",
            num_samples,
            "num of repaired nodes",
            num_nodes,
            "repair node list",
            repair_node_list,
            "repair layer",
            layer_to_repair,
            "Num of nodes in repair layer",
            32,
            "timelimit",
            options.optimizer_options["timelimit"],
            "mipfocus",
            options.optimizer_options["mipfocus"],
            "max weight bunds",
            cost_weights,
            "cost weights",
            cost_weights,
            "output bounds",
            output_bounds,
        ]
        # Add contents of list as last row in the csv file
        csv_writer.writerow(model_evaluation)
    print("saved: stats")

    plotTestData(
        ctrl_model_orig,
        out_model,
        train_obs,
        train_ctrls,
        test_obs,
        test_ctrls,
        x_train,
        y_train,
        now_str,
        3,
    )

    pass
