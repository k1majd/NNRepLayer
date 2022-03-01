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

from numpy import sin, cos, pi
from shapely.geometry import Polygon
from tensorflow import keras
from pyomo.gdp import *

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
2) modification to repair weights (input architecture): tf and pytorch,
    future (h5py discussion)
3) Defined cost expression: MSE, SSE
4) Break model into two parts: 
 

    1)   f(lin cons, model, samples)
            ...
            return solver_model, variable
        
    2) define additional constraints on solver_model and variables
    3) run wrapper, pyomo



'''


from matplotlib.pyplot import text
# from matplotlib import pyplot as plt
# plt.rcParams['text.usetex'] = True
import pickle 
import matplotlib as mpl
import numpy as np

# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_predict_train.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_orig = pickle.load(filehandle)

# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_new_last_layer.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_4 = pickle.load(filehandle)

# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_new_third_layer.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_3 = pickle.load(filehandle)

# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_train.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_train = pickle.load(filehandle)
    

# def csa(y_pred_train_orig):
#     miss_samples_orig = np.sum(y_pred_train_orig[:,0]>0.5)
#     print(miss_samples_orig)
#     return (y_pred_train_orig.shape[0]-miss_samples_orig)/y_pred_train_orig.shape[0]

# csa_orig_train = csa(y_orig)
# csa_rep_4_train = csa(y_4)
# csa_rep_3_train = csa(y_3)


# print(csa_orig_train)
# print(csa_rep_4_train)
# print(csa_rep_3_train)


# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_predict_train.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_orig = pickle.load(filehandle)

# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_new_last_layer.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_4 = pickle.load(filehandle)

# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_new_third_layer.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_3 = pickle.load(filehandle)

# with open('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/fk_data/y_train.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     y_train = pickle.load(filehandle)



transform1 = np.array([[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]])  # transformation matrix 1
transform2 = np.array([[1, 0, -2.5], [0, 1, -2.5], [0, 0, 1]])  # transformation matrix 2
rotate = np.array([[cos(pi / 4), -sin(pi / 4), 0], [sin(pi / 4), cos(pi / 4), 0], [0, 0, 1]])  # rotation matrix
inp = np.array([[1.25, 3.75, 3.75, 1.25],[1.25, 1.25, 3.75, 3.75],[1, 1 , 1, 1]])
out = np.matmul(np.matmul(np.matmul(transform1, rotate), transform2), inp)
poly3 = Polygon([(out[0, 0], out[1, 0]), (out[0, 1], out[1, 1]), (out[0, 2], out[1, 2]), (out[0, 3], out[1, 3])])


# plt.plot(x_poly3, y_poly3, color='green', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2, label='Target Set')
# plt.legend(loc="upper left")
# plt.show()

model_orig = keras.models.load_model('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/affine_transform_model')
rep_model_3 = keras.models.load_model('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/repaired_affine_transform_layer3')
rep_model_2 = keras.models.load_model('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/repaired_affine_transform_layer2')

with open("/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/io_data_affine_transform.pickle", "rb") as data:
    x_train, y_train, x_test, y_test = pickle.load(data)

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

y_test_orig = model_orig.predict(x_test)
y_test_rep_3 = rep_model_3.predict(x_test)
y_test_rep_2 = rep_model_2.predict(x_test)


def csa(y_pred_train_orig):
    miss_samples_orig = 0
    for i in range(y_pred_train_orig.shape[0]):
        if not poly3.contains(Point(y_pred_train_orig[i,:])):
            miss_samples_orig = miss_samples_orig + 1
    # print(miss_samples_orig)
    return (y_pred_train_orig.shape[0]-miss_samples_orig)/y_pred_train_orig.shape[0]

csa_orig_train = csa(y_test_orig)
csa_rep_3_train = csa(y_test_rep_3)
csa_rep_2_train = csa(y_test_rep_2)



print(csa_orig_train)
print(csa_rep_3_train)
print(csa_rep_2_train)
