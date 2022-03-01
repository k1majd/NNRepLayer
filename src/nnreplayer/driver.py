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

import pickle
from tensorflow import keras
import tensorflow.keras.backend as kb


'''
Done:
1) Function to repair weights: outputs new parameters and nkeras model with new weights
2) Testing and verifcation of results, bug fixes and basic test cases for sanity checks.
3) Cost expression as input to weight_repair function.

In Progress: 
1) Function to the gurobi for output constraints
2) Test for both the examples and do sanity checks
3) Write sampler for NN
4) Check for cost expression parsing and standardize the input, ouputs

Remaining:
1) Store in LP format
2) modification to repair weight (input architecture): tf and pytorch,
    future (h5py discussion)
3) Defined cost expression: MSE, SSE
4) Break model into two parts: 
 

    1)   f(lin cons, model, samples)
            ...
            return solver_model, variable
        
    2) define additional constraints on solver_model and variables
    3) run wrapper, pyomo



'''
#define costum loss function
def keras_SSE_costum(y_actual, y_predicted):
    err = y_actual - y_predicted
    loss_value = kb.sum(kb.square(err))
    return loss_value

model_orig = keras.models.load_model('/home/daittan/NNRepLayer/nnreplayer/repaired_affine_transform_layer3')


num_input = 3
num_output = 3
#num_layers_0 = 3
num_hidden_0 = 20
num_hidden_1 = 20
architecture = [num_input, num_hidden_0, num_hidden_1, num_output]



with open("/home/daittan/NNRepLayer/nnreplayer/io_data_affine_transform.pickle", "rb") as data:
    x_train, y_train, x_test, y_test = pickle.load(data)

# x_train = np.transpose(np.array([x_train]))
# x_test = np.transpose(np.array([x_test]))
architecture = [num_input, num_hidden_0, num_hidden_1, num_output]


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




#########################################
# Form
# ----------------
# f(y_train, y_predict):
#   cost = ...
#   return cost


# Create options for user
# -----------------------
# sum squared error
# mean square error

def squared_sum(x, y):
    m,n = np.array(x).shape
    _squared_sum = 0
    for i in range(m):
        for j in range(n):
            _squared_sum += (x[i, j] - y[i, j]) ** 2
    return _squared_sum
#########################################

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

from .utils.options import Options
from .repair.perform_repair import perform_repair
layer_to_repair = 3
train_dataset = (x_train, y_train)
options = Options('gdp.bigm', 'gurobi', "python", "keras", 100)

results = perform_repair(layer_to_repair, model_orig, architecture, A,b, squared_sum, train_dataset, options)


y_new_train = results.new_model.predict(x_train)
y_new_test = results.new_model.predict(x_test)


y_train_original = model_orig.predict(x_train)
y_test_original = model_orig.predict(x_test)

print("weight_error: {}".format(results.weight_error))
print("bias_error: {}".format(results.bias_error))

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
