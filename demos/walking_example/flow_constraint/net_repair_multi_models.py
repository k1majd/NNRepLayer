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
    box,
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
    # fig = plt.figure(figsize=(13, 5))
    # gs = fig.add_gridspec(2, 1)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[1, 0])
    ax1.quiver(
        x,
        y,
        x_vel,
        y_vel,
        angles="xy",
        scale_units="xy",
        scale=1,
        alpha=0.5,
    )
    ax1.grid(alpha=0.5, linestyle="dashed")
    ax1.set_xlabel(var1 + " angle")
    ax1.set_ylabel(var2 + " angle")
    plt.xlim([-3.0, 3.0])
    plt.ylim([-10.0, 5.0])
    # plot box
    if box:
        ax1.plot(
            [box[0], box[0]], [box[2], box[3]], "r--", linewidth=1.5, alpha=0.5
        )
        ax1.plot(
            [box[1], box[1]], [box[2], box[3]], "r--", linewidth=1.5, alpha=0.5
        )
        ax1.plot(
            [box[0], box[1]], [box[2], box[2]], "r--", linewidth=1.5, alpha=0.5
        )
        ax1.plot(
            [box[0], box[1]], [box[3], box[3]], "r--", linewidth=1.5, alpha=0.5
        )

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

    plt.show()


def is_in_box(point, box):
    return box[0] <= point[0] <= box[1] and box[2] <= point[1] <= box[3]


def generate_repair_dataset(obs, ctrl, model, num_samples, box, dt=1.0):
    x_train = []
    y_train = []
    samples_alread = 0
    pred_ctrl = model.predict(obs)
    for i in range(obs.shape[0]):
        point = np.array(
            [obs[i, -5] + obs[i, -4], obs[i, -1] + pred_ctrl[i][0]]
        )
        if is_in_box(point, box):
            x_train.append(obs[i])
            y_train.append(ctrl[i])
        else:
            if (samples_alread < 100) and (np.random.randint(2) == 1):
                point = np.array([obs[i, -5], obs[i, -1]])
                if is_in_box(point, [-2.0, -0.5, 1.0, 2.0]):
                    x_train.append(obs[i])
                    y_train.append(ctrl[i])
                    samples_alread += 1
    return np.array(x_train), np.array(y_train)


def generate_repair_dataset2(obs, ctrl, model, num_samples, box, dt=1.0):
    # x_train = []
    # y_train = []
    samples_alread = 0
    pred_ctrl = model.predict(obs)
    idx_adv = []
    idx_no_adv = []
    for i in range(obs.shape[0]):
        point = np.array(
            [obs[i, -5] + obs[i, -4], obs[i, -1] + pred_ctrl[i][0]]
        )
        if is_in_box(point, box):
            idx_adv.append(i)
        else:
            point = np.array([obs[i, -5], obs[i, -1]])
            if is_in_box(point, [-2.0, -0.5, 1.0, 2.0]):
                idx_adv.append(i)
            else:
                idx_no_adv.append(i)
    id_train_adv = np.random.choice(
        idx_adv, int(0.50 * num_samples), replace=False
    )
    id_train_no_adv = np.random.choice(
        idx_no_adv, int(0.50 * num_samples), replace=False
    )
    id_train = np.concatenate((id_train_adv, id_train_no_adv))
    x_train = obs[id_train]
    y_train = ctrl[id_train]
    return x_train, y_train


if __name__ == "__main__":
    now = datetime.now()
    now_str = f"_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}_{now.second}"
    # load model
    box = [-2.5, -0.5, 0.5, 3.5]  # xmin,xmax,ymin,ymax
    ctrl_model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__)) + "/models/model_orig"
    )
    # Train window
    num_samples = 1000
    (
        train_obs,
        train_ctrls,
        x_test,
        y_test,
    ) = generateDataWindow(10)
    x_train, y_train = generate_repair_dataset2(
        train_obs, train_ctrls, ctrl_model_orig, num_samples, box
    )
    # rnd_pts = np.random.choice(x_train.shape[0], 200)
    # x_train = x_train[rnd_pts, :]
    # y_train = y_train[rnd_pts]
    plot_pahse(
        ctrl_model_orig,
        x_train,
        y_train,
        "femur",
        "ankle",
        box,
    )

    # rnd_pts = np.random.choice(1000, num_samples)

    # x_train = test_obs[0:1, :]
    # y_train = test_ctrls[0:1]

    # plotTestData(
    #     ctrl_model_orig,
    #     train_obs,
    #     train_ctrls,
    #     test_obs,
    #     test_ctrls,
    #     now_str,
    # )

    repair_obj = NNRepair(ctrl_model_orig)

    # Define Constraint

    # def constraint_outside(model, i):
    #     return [
    #         [
    #             (x_train[i, -1] + getattr(model, repair_obj.output_name)[i, 0])
    #             - box[0]
    #             <= 0
    #         ],
    #         [
    #             box[1]
    #             - (
    #                 x_train[i, -1]
    #                 + getattr(model, repair_obj.output_name)[i, 0]
    #             )
    #             <= 0
    #         ],
    #         [
    #             getattr(model, repair_obj.output_name)[i, 0] - box[0] <= 0,
    #             box[0] - getattr(model, repair_obj.output_name)[i, 0] <= 0,
    #             box[3] - getattr(model, repair_obj.output_name)[i, 1] <= 0,
    #         ],
    #         [
    #             getattr(model, repair_obj.output_name)[i, 0] - box[1] <= 0,
    #             box[0] - getattr(model, repair_obj.output_name)[i, 0] <= 0,
    #             getattr(model, repair_obj.output_name)[i, 1] - box[2] <= 0,
    #         ],
    #     ]
    y_pred_ctrl = ctrl_model_orig.predict(x_train)

    def constraint_outside(model, i):
        point1 = np.array(
            [
                x_train[i, -5] + x_train[i, -4],
                x_train[i, -1] + y_pred_ctrl[i][0],
            ]
        )
        point2 = np.array([x_train[i, -5], x_train[i, -1]])
        if is_in_box(point1, box) or is_in_box(point2, [-2.0, -0.5, 1.0, 2.0]):
            return (
                x_train[i, -1] + getattr(model, repair_obj.output_name)[i, 0]
                <= 0.9
            )
        else:
            return pyo.Constraint.Skip

    ######

    layer_to_repair = 3  # first layer-(0) last layer-(4)
    max_weight_bound = 2.0  # specifying the upper bound of weights error
    cost_weights = np.array([10.0, 1.0])  # cost weights
    # output_bounds = (-30.0, 50.0)
    repair_node_list = []
    num_nodes = len(repair_node_list) if len(repair_node_list) != 0 else 32
    w_error_norm = 0
    repair_obj.compile(
        x_train,
        y_train,
        layer_to_repair,
        # output_constraint_list=output_constraint_list,
        cost_weights=cost_weights,
        max_weight_bound=max_weight_bound,
        data_precision=6,
        param_precision=6,
        # repair_node_list=repair_set,
        repair_node_list=repair_node_list,
        w_error_norm=w_error_norm,
        # output_bounds=output_bounds,
    )
    setattr(
        repair_obj.opt_model,
        "constraint_outside" + str(layer_to_repair),
        pyo.Constraint(range(repair_obj.num_samples), rule=constraint_outside),
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
            "timelimit": 18000,  # max time algorithm will take in seconds
            "mipgap": 0.01,  #
            "mipfocus": 2,  #
            "cuts": 0,
            "concurrentmip": 3,
            "threads": 48,
            "improvestarttime": 16000,
            "cuts": 0,
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
        path_write + f"/models/model_layer{now_str}",
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
        pickle.dump([x_train, y_train, x_test, y_test], data)

    # save summary
    pred_ctrls = out_model(x_test, training=False)
    err = np.abs(y_test - pred_ctrls)
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
            # "output bounds",
            # output_bounds,
        ]
        # Add contents of list as last row in the csv file
        csv_writer.writerow(model_evaluation)
    print("saved: stats")
    plot_pahse(
        out_model,
        train_obs,
        train_ctrls,
        "femur",
        "ankle",
        box,
    )

    # plotTestData(
    #     ctrl_model_orig,
    #     out_model,
    #     x_test,
    #     y_test,
    #     now_str,
    #     bound,
    #     layer_to_repair,
    # )
