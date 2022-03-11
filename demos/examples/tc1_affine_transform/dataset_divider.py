import os
import pickle
import numpy as np
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt


direc = os.path.dirname(os.path.realpath(__file__))
path = direc + "/tc1/original_net"
if not os.path.exists(path + "/data/input_output_data_tc1.pickle"):
    raise ImportError(
        "path {path}/data/input_output_data_tc1.pickle does not exist!"
    )
with open(path + "/data/input_output_data_tc1.pickle", "rb") as data:
    dataset = pickle.load(data)

## affine transformation matrices
translate1 = np.array(
    [[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]]
)  # translation matrix 1
translate2 = np.array(
    [[1, 0, -2.5], [0, 1, -2.5], [0, 0, 1]]
)  # translation matrix 2
rotate = np.array(
    [
        [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
        [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
        [0, 0, 1],
    ]
)  # rotation matrix

## original, transformed, and constraint Polygons
poly_orig = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
poly_trans = Polygon(
    [(2.5, 4.621), (4.624, 2.5), (2.5, 0.3787), (0.3787, 2.5)]
)
inp_const_vertices = np.array(
    [[1.25, 3.75, 3.75, 1.25], [1.25, 1.25, 3.75, 3.75], [1, 1, 1, 1]]
)  # contraint vertices in input space
out_const_vertices = np.matmul(
    np.matmul(np.matmul(translate1, rotate), translate2), inp_const_vertices
)  # constraint vertices in output space
poly_const = Polygon(
    [
        (out_const_vertices[0, 0], out_const_vertices[1, 0]),
        (out_const_vertices[0, 1], out_const_vertices[1, 1]),
        (out_const_vertices[0, 2], out_const_vertices[1, 2]),
        (out_const_vertices[0, 3], out_const_vertices[1, 3]),
    ]
)

# divide training dataset
x_inside = []
x_outside = []
y_inside = []
y_outside = []

for i in range(dataset[0].shape[0]):
    if Point([dataset[1][i][0], dataset[1][i][1]]).within(poly_const):
        x_inside.append(dataset[0][i])
        y_inside.append(dataset[1][i])
    else:
        x_outside.append(dataset[0][i])
        y_outside.append(dataset[1][i])

if not os.path.exists(path + "/data"):
    os.makedirs(path + "/data")
with open(
    path + "/data/input_output_data_inside_train_tc1.pickle", "wb"
) as data:
    print(f"number of training points inside: {len(y_inside)}")
    pickle.dump([np.array(x_inside), np.array(y_inside)], data)
with open(
    path + "/data/input_output_data_outside_train_tc1.pickle", "wb"
) as data:
    print(f"number of training points outside: {len(y_outside)}")
    pickle.dump([np.array(x_outside), np.array(y_outside)], data)

# divide testing dataset
x_inside = []
x_outside = []
y_inside = []
y_outside = []

for i in range(dataset[2].shape[0]):
    if Point([dataset[3][i][0], dataset[3][i][1]]).within(poly_const):
        x_inside.append(dataset[2][i])
        y_inside.append(dataset[3][i])
    else:
        x_outside.append(dataset[2][i])
        y_outside.append(dataset[3][i])
if not os.path.exists(path + "/data"):
    os.makedirs(path + "/data")
with open(
    path + "/data/input_output_data_inside_test_tc1.pickle", "wb"
) as data:
    print(f"number of testing points inside: {len(y_inside)}")
    pickle.dump([np.array(x_inside), np.array(y_inside)], data)
with open(
    path + "/data/input_output_data_outside_test_tc1.pickle", "wb"
) as data:
    print(f"number of testing points outside: {len(y_outside)}")
    pickle.dump([np.array(x_outside), np.array(y_outside)], data)
