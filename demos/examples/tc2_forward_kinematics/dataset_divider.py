import os
import pickle
import numpy as np
from matplotlib import pyplot as plt


direc = os.path.dirname(os.path.realpath(__file__))
path = direc + "/tc2/original_net"
if not os.path.exists(path + "/data/input_output_data_tc2.pickle"):
    raise ImportError(
        "path {path}/data/input_output_data_tc2.pickle does not exist!"
    )
with open(path + "/data/input_output_data_tc2.pickle", "rb") as data:
    dataset = pickle.load(data)

# constraint
A = np.array([[1.0, 0.0, 0.0, 0.0]])
b = 0.5

# divide training dataset
x_inside = []
x_outside = []
y_inside = []
y_outside = []

for i in range(dataset[0].shape[0]):
    if np.matmul(A, dataset[1][i]) <= b:
        x_inside.append(dataset[0][i])
        y_inside.append(dataset[1][i])
    else:
        x_outside.append(dataset[0][i])
        y_outside.append(dataset[1][i])

if not os.path.exists(path + "/data"):
    os.makedirs(path + "/data")
with open(
    path + "/data/input_output_data_inside_train_tc2.pickle", "wb"
) as data:
    print(f"number of training points inside: {len(y_inside)}")
    pickle.dump([np.array(x_inside), np.array(y_inside)], data)
with open(
    path + "/data/input_output_data_outside_train_tc2.pickle", "wb"
) as data:
    print(f"number of training points outside: {len(y_outside)}")
    pickle.dump([np.array(x_outside), np.array(y_outside)], data)

# divide testing dataset
x_inside = []
x_outside = []
y_inside = []
y_outside = []

for i in range(dataset[2].shape[0]):
    if np.matmul(A, dataset[3][i]) <= b:
        x_inside.append(dataset[2][i])
        y_inside.append(dataset[3][i])
    else:
        x_outside.append(dataset[2][i])
        y_outside.append(dataset[3][i])
if not os.path.exists(path + "/data"):
    os.makedirs(path + "/data")
with open(
    path + "/data/input_output_data_inside_test_tc2.pickle", "wb"
) as data:
    print(f"number of testing points inside: {len(y_inside)}")
    pickle.dump([np.array(x_inside), np.array(y_inside)], data)
with open(
    path + "/data/input_output_data_outside_test_tc2.pickle", "wb"
) as data:
    print(f"number of testing points outside: {len(y_outside)}")
    pickle.dump([np.array(x_outside), np.array(y_outside)], data)
