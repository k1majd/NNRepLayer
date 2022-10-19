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
        data = np.asarray(list(csv.reader(csv_file, delimiter=',')), dtype=np.float32)
    return data


def squared_sum(x, y):
    m,n = np.array(x).shape
    _squared_sum = 0
    for i in range(m):
        for j in range(n):
            _squared_sum += (x[i, j] - y[i, j]) ** 2
    return _squared_sum


def generateDataWindow(window_size):
    Dfem = loadData('demos/walking_example/data/GeoffFTF_1.csv')
    Dtib = loadData('demos/walking_example/data/GeoffFTF_2.csv')
    Dfut = loadData('demos/walking_example/data/GeoffFTF_3.csv')
    phase = loadData('demos/walking_example/data/GeoffFTF_phase.csv')
    n=20364
    Dankle = np.subtract(Dtib[:n,1], Dfut[:n,1])
    observations = np.concatenate((Dfem[:n,1:], Dtib[:n,1:]), axis=1)
    observations = (observations - observations.mean(0))/observations.std(0)
    controls = Dankle  #(Dankle - Dankle.mean(0))/Dankle.std(0)
    phase = phase[:n]
    n_train = 19000
    train_observation = np.array([]).reshape(0,4*window_size+1)
    test_observation = np.array([]).reshape(0,4*window_size+1)

    for i in range(n_train):
        temp_obs = np.array([]).reshape(1,0)
        for j in range(window_size):
            temp_obs = np.concatenate((temp_obs, observations[i+j,:].reshape(1,-1)), axis=1)
        temp_obs = np.concatenate((temp_obs, 0*phase[i+window_size].reshape(1,1)), axis=1)
        train_observation = np.concatenate((train_observation, temp_obs),axis=0)
    train_controls = controls[window_size:n_train+window_size].reshape(-1,1)

    for i in range(n_train,n-window_size):
        temp_obs = np.array([]).reshape(1,0)
        for j in range(window_size):
            temp_obs = np.concatenate((temp_obs, observations[i+j,:].reshape(1,-1)), axis=1)
        temp_obs = np.concatenate((temp_obs, 0*phase[i+window_size].reshape(1,1)), axis=1)
        test_observation = np.concatenate((test_observation, temp_obs),axis=0)
    test_controls = controls[n_train+window_size:].reshape(-1,1)

    return train_observation, train_controls, test_observation, test_controls


def buildModelWindow(data_size):
    input_size = data_size[0]
    layer_size1 = 32
    layer_size2 = 32
    layer_size3 = 32
    output_size = 1

    model = keras.Sequential([
        layers.Dense(layer_size1, activation=tf.nn.relu, input_shape=[input_size]),
        layers.Dense(layer_size2, activation=tf.nn.relu),
        layers.Dense(layer_size3, activation=tf.nn.relu),
        layers.Dense(output_size)
    ])
    model.compile(optimizer='adam', loss=[tf.keras.losses.MeanAbsoluteError()], metrics=["accuracy"])
    model.summary()
    architecture = [input_size, layer_size1, layer_size2, layer_size3, output_size]
    filepath = 'models/model1'
    tf_callback=[tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weight_only=False, mode='auto', save_freq='epoch', options=None), keras.callbacks.TensorBoard(log_dir='tf_logs')]
    return model, tf_callback, architecture


def plotTestData(model, train_obs, train_ctrls, test_obs, test_ctrls):
    pred_ctrls = model(test_obs, training=False)

    plt.figure(1)
    plt.plot(test_ctrls, color='#173f5f')
    plt.plot(pred_ctrls, color=[.4705, .7921, .6470])
    plt.grid(alpha=.5, linestyle='dashed')
    # plt.xlim([600,900])
    plt.ylabel('Ankle Angle Control (rad)')
    plt.xlabel('Time (s)')
    plt.show()
    # plt.savefig("../figures/layer4_60_n60.pdf")



if __name__ == "__main__":
    # Train window model
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)
    ctrl_model_orig, callback, architecture = buildModelWindow(train_obs[0].shape)
    ctrl_model_orig.load_weights('models/model1')
    # ctrl_model_orig.fit(train_obs, train_ctrls, validation_data=(test_obs,test_ctrls), batch_size=8, epochs=100, use_multiprocessing=True, verbose=1, shuffle = False, callbacks=callback)
    plotTestData(ctrl_model_orig, train_obs, train_ctrls, test_obs, test_ctrls)

    bound_upper = 24
    bound_lower = 14
    A = np.array([[1],[-1]])
    b = np.array([[bound_upper],[bound_lower]])

    # input the constraint list
    constraint_inside = ConstraintsClass("inside", A, b) # ConstraintsClass(A, b, C, d)
    output_constraint_list = [constraint_inside]
    repair_obj = NNRepair(ctrl_model_orig)

    layer_to_repair = 4    # first layer-(0) last layer-(4)
    max_weight_bound = 10   # specifying the upper bound of weights error
    cost_weights = np.array([1.0, 1.0]) # cost weights
    repair_obj.compile(
        test_obs,
        test_ctrls,
        layer_to_repair,
        output_constraint_list=output_constraint_list,
        cost_weights=cost_weights,
        max_weight_bound=max_weight_bound,
        output_bounds=(-1e3, 1e3),
    )


    direc = os.getcwd()
    path_write = os.path.join(direc, "repair_net")

    # check directories existence
    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print(f"Directory: {path_write} is created!")

    # setup directory to store optimizer log file
    if not os.path.exists(path_write + "/logs"):
        os.makedirs(path_write + "/logs")

    # setup directory to store the modeled MIP and parameters
    if not os.path.exists(path_write + "/summary"):
        os.makedirs(path_write + "/summary")

    # setup directory to store the repaired model
    if not os.path.exists(path_write):
        os.makedirs(path_write + f"/model_layer_{layer_to_repair}")

    # specify options
    options = Options(
        "gdp.bigm",
        "gurobi",
        "python",
        "keras",
        {
            "timelimit": 3600,  # max time algorithm will take in seconds
            "mipgap": 0.0001,    # 
            "mipfocus": 1,      # 1, 2, 3 for different optimization stratagies
            "improvestarttime": 3000,
            "logfile": path_write
            + f"/logs/opt_log_layer{layer_to_repair}.log",
        },
    )

    # repair the network
    out_model = repair_obj.repair(options)
    plotTestData(out_model, train_obs, train_ctrls, test_obs, test_ctrls)

