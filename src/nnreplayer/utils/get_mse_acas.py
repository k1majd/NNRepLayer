import numpy as np
from tensorflow import keras
from pyomo.gdp import *


import pickle
from tensorflow import keras
from acasxu_tf_keras.gen_tf_keras import read_acasxu_weights, build_dnn1

nnet_path = "nnet_2_9.nnet"
model_orig = build_dnn1(read_acasxu_weights(nnet_path))

with open("/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/input_output_data_acas.pickle", "rb") as data:
    x_train, y_train, x_test, y_test = pickle.load(data)


def squared_sum(x, y):
    m,n = np.array(x).shape
    _squared_sum = 0
    for i in range(m):
        for j in range(n):
            _squared_sum += (x[i, j] - y[i, j]) ** 2
    return _squared_sum


rep_model = keras.models.load_model('/home/local/ASUAD/tkhandai/nn_repair/NN-Repair/repaired_acas_xu_layer7')
w_rep = rep_model.layers[-2].kernel.numpy()
b_rep = rep_model.layers[-2].bias.numpy()
w_orig = model_orig.layers[-2].kernel.numpy()
b_orig = model_orig.layers[-2].bias.numpy()

weight_error = np.max(w_rep - w_orig)
bias_error = np.max(b_rep - b_orig)
print("weight_error: {}".format(weight_error))
print("bias_error: {}".format(bias_error))

y_pred_train_orig = model_orig.predict(x_train)
y_pred_train_rep = rep_model.predict(x_train)

y_pred_test_orig = model_orig.predict(x_test)
y_pred_test_rep = rep_model.predict(x_test)

def csa(y_pred_train_orig):
    miss_samples_orig = np.sum(np.argmin(y_pred_train_orig,1)==3)
    print(miss_samples_orig)
    return (y_pred_train_orig.shape[0]-miss_samples_orig)/y_pred_train_orig.shape[0]

csa_orig_train = csa(y_pred_test_rep)
print(csa_orig_train)
# # a = np.mean((y_train-y_pred_train_orig)**2)
# # b = np.mean((y_test-y_pred_test_orig)**2)
# a = np.mean((y_train-y_pred_train_rep)**2)
# b = np.mean((y_test-y_pred_test_rep)**2)
# # c = np.mean((y_pred_train_rep-y_pred_train_orig)**2)
# # d = np.mean((y_pred_test_rep-y_pred_test_orig)**2)

# print("{} {}".format(a,b))