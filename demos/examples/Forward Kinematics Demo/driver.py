import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
from pprint import pprint
from numpy import sin, cos, pi
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from tensorflow import keras
from pyomo.gdp import *
from scipy.spatial import ConvexHull
import os
import pickle
from tensorflow import keras
import tensorflow.keras.backend as kb

from nnreplayer.utils.options import Options
from nnreplayer.utils.utils import constraints_class
from nnreplayer.repair.perform_repair import perform_repair


#define costum loss function
def keras_SSE_costum(y_actual, y_predicted):
    err = y_actual - y_predicted
    loss_value = kb.sum(kb.square(err))
    return loss_value

model_orig = keras.models.load_model(os.getcwd()+'/forward_kinematics_original_model', compile= False)





with open(os.getcwd() + "/forward_kinematics_io_dataset.pickle", "rb") as data:
    x_train, y_train, x_test, y_test = pickle.load(data)

x_train = np.reshape(x_train, (-1,1))
x_test = np.reshape(x_test, (-1,1))

num_input = 1
num_output = 2
num_hidden_0 = 10
num_hidden_1 = 10
architecture = [num_input, num_hidden_0, num_hidden_1, num_output]

def squared_sum(x, y):
    m,n = np.array(x).shape
    _squared_sum = 0
    for i in range(m):
        for j in range(n):
            _squared_sum += (x[i, j] - y[i, j]) ** 2
    return _squared_sum
    
A_1 = np.array([[1,0],[-1,0],[1,0],[1,0]])
b_1 = np.array([[0.45],[-0.55],[0.55],[0.55]])

A_2 = np.array([[0, 0], [0, 0],[-1,0],[-1,0]])
b_2 = np.array([[0],[0],[-0.45],[-0.45]])

A_3 = np.array([[0, 0], [0, 0],[0,-1],[0,1]])
b_3 = np.array([[0],[0],[-0.25],[0.1]])

A = [A_1, A_2, A_3]
B = [b_1,b_2,b_3]
constraint_outside = constraints_class("outside", A, B)

output_constraint_list = [constraint_outside]


layer_to_repair = 3
train_dataset = (x_train, y_train)
options = Options('gdp.bigm', 'gurobi', "python", "keras", 100)

results = perform_repair(layer_to_repair, model_orig, architecture, output_constraint_list, squared_sum, train_dataset, options)


y_new_train = results.new_model.predict(x_train)
y_new_test = results.new_model.predict(x_test)


y_train_original = model_orig.predict(x_train)
y_test_original = model_orig.predict(x_test)

print("weight_error: {}".format(results.weight_error))
print("bias_error: {}".format(results.bias_error))

