#!/usr/bin/env python
# coding: utf-8


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
import numpy as np

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


# # Introduction
# 
# This demo is for repairing a neural network that repairs the affine transform neural network example to follow a certain constraint. We perform the folowing operations here:
# 
# 1) Load the neural network and samples that have both normal and adverserial examples.
# 2) Define the constraints
# 3) Define the layer to repair and run the repair

# # Load Neural Network


#define costum loss function
def keras_SSE_costum(y_actual, y_predicted):
    err = y_actual - y_predicted
    loss_value = kb.sum(kb.square(err))
    return loss_value

model_orig = keras.models.load_model(os.getcwd()+'/affine_transform_original_model', compile= False)





with open(os.getcwd() + "/affine_io_dataset.pickle", "rb") as data:
    x_train, y_train, x_test, y_test = pickle.load(data)




num_input = 3
num_output = 3
num_hidden_0 = 20
num_hidden_1 = 10
architecture = [num_input, num_hidden_0, num_hidden_1, num_output]
architecture = [num_input, num_hidden_0, num_hidden_1, num_output]

def squared_sum(x, y):
    m,n = np.array(x).shape
    _squared_sum = 0
    for i in range(m):
        for j in range(n):
            _squared_sum += (x[i, j] - y[i, j]) ** 2
    return _squared_sum
    



transform1 = np.array([[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]])  # transformation matrix 1
transform2 = np.array([[1, 0, -2.5], [0, 1, -2.5], [0, 0, 1]])  # transformation matrix 2
rotate = np.array([[cos(pi / 4), -sin(pi / 4), 0], [sin(pi / 4), cos(pi / 4), 0], [0, 0, 1]])  # rotation matrix
inp = np.array([[1.25, 3.75, 3.75, 1.25],[1.25, 1.25, 3.75, 3.75],[1, 1 , 1, 1]])
out = np.matmul(np.matmul(np.matmul(transform1, rotate), transform2), inp)
poly3 = Polygon([(out[0, 0], out[1, 0]), (out[0, 1], out[1, 1]), (out[0, 2], out[1, 2]), (out[0, 3], out[1, 3])])

# get the coordinates of the exterior points of the polytope
ex_points = np.array(poly3.exterior.coords)

# get A and b matrices
hull = ConvexHull(ex_points)
eqs = np.array(hull.equations)
A = eqs[0:eqs.shape[0],0:eqs.shape[1]-1]
b = -eqs[0:eqs.shape[0],-1]

b = np.array([b]).T



poly_outside = Polygon([(2,2),(2,3),(3,3),(3,2)])

x_poly_out, y_poly_out = poly_outside.exterior.xy

A_1 = np.array([[1,0],[-1,0],[1,0],[1,0]])
b_1 = np.array([[2],[-3],[3],[3]])

A_2 = np.array([[0, 0], [0, 0],[-1,0],[-1,0]])
b_2 = np.array([[0],[0],[-2],[-2]])

A_3 = np.array([[0, 0], [0, 0],[0,-1],[0,1]])
b_3 = np.array([[0],[0],[-3],[2]])

A_out = [A_1, A_2, A_3]
B_out = [b_1,b_2,b_3]



constraint_inside = constraints_class("inside", A, b)
constraint_outside = constrainsts_class("outside", A_out, B_out)
output_constraint_list = [constraint_inside, constraint_outside]


# # Run the Repair Problem


layer_to_repair = 3
train_dataset = (x_train, y_train)
options = Options('gdp.bigm', 'gurobi', "python", "keras", 100, 3600, 0.01)

results = perform_repair(layer_to_repair, model_orig, architecture, output_constraint_list, squared_sum, train_dataset, options)


y_new_train = results.new_model.predict(x_train)
y_new_test = results.new_model.predict(x_test)


y_train_original = model_orig.predict(x_train)
y_test_original = model_orig.predict(x_test)

print("weight_error: {}".format(results.weight_error))
print("bias_error: {}".format(results.bias_error))


# # Plot the Outputs


def plot_model(model, x_true, y_true, arg):
    y_predict = model.predict(x_true)
    
    ## training output
    plt.plot(y_true[:,0], y_true[:,1], 'ro', label='Original Samples')

    ## predicted output
    plt.plot(y_predict[:,0], y_predict[:,1], 'ko', label='Predicted Output')
    plt.title(arg)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="upper left")
    plt.show()
    return y_predict


num_pts = 200
## polygon vertices
poly = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
poly2 = Polygon([(2.5, 4.621), (4.624, 2.5), (2.5, 0.3787), (0.3787, 2.5)])  # transformed polygon vertices
x_poly, y_poly = poly.exterior.xy
x_poly2, y_poly2 = poly2.exterior.xy
x_poly3, y_poly3 = poly3.exterior.xy
print(x_train.shape)
plt.plot(x_train[:,0], x_train[:,1], 'bo', label='Training samples')
plt.plot(y_train_original[:,0], y_train_original[:,1], 'ro', label='Original Model ouput')
plt.plot(x_poly, y_poly, color='blue', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, label='Input Set')
plt.plot(x_poly2, y_poly2, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, label='Target Set')
plt.plot(x_poly3, y_poly3, color='green', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, label='Target Set')
plt.plot(y_new_train[:,0], y_new_train[:,1], 'yo', label='Repaired Model Ouptut')
plt.legend(loc="upper left")
plt.show()

num_pts = 200
## polygon vertices
poly = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
poly2 = Polygon([(2.5, 4.621), (4.624, 2.5), (2.5, 0.3787), (0.3787, 2.5)])  # transformed polygon vertices
x_poly, y_poly = poly.exterior.xy
x_poly2, y_poly2 = poly2.exterior.xy
x_poly3, y_poly3 = poly3.exterior.xy
print(x_train.shape)
plt.plot(x_test[:,0], x_test[:,1], 'bo', label='Training samples')
plt.plot(y_test_original[:,0], y_test_original[:,1], 'ro', label='Original Model ouput')
plt.plot(x_poly, y_poly, color='blue', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, label='Input Set')
plt.plot(x_poly2, y_poly2, color='red', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, label='Target Set')
plt.plot(x_poly3, y_poly3, color='green', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, label='Target Set')
plt.plot(y_new_test[:,0], y_new_test[:,1], 'yo', label='Repaired Model Ouptut')
plt.legend(loc="upper left")
plt.show()

results.new_model.save("repaired_affine_transform_layer{}".format(layer_to_repair), '/home/daittan/NN-Repair/NN-Repair/')



