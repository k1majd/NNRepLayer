from shapely.geometry import Polygon, Point
import numpy as np
from sklearn.model_selection import train_test_split
import os

## batch creation
class Batch(object):
    def __init__(self, X_col, Y_col, batch_size_val):
        self.X = X_col
        self.Y = Y_col
        self.size = X_col.shape[0]
        self.train_size = batch_size_val
        self.test_size = self.size - batch_size_val

    def getBatch(self):
        values = range(self.size)
        train_dataset, test_dataset = train_test_split(
            values, train_size=self.train_size, test_size=self.test_size
        )
        return (
            self.X[train_dataset, :],
            self.Y[train_dataset, :],
            self.X[test_dataset, :],
            self.Y[test_dataset, :],
        )


## reshape input
def reshape_cos_sin(inp):
    row = inp.shape[0]
    col = inp.shape[1]
    x = np.zeros((row, 2 * col))
    for i in range(row):
        for j in range(col):
            x[i, 2 * j] = np.cos(inp[i, j])
            x[i, 2 * j + 1] = np.sin(inp[i, j])
    return x


## generate data samples
def data_generate(num_pts, unif2edge=0.75, edge_scale=0.7):

    num_pts_unif = int(unif2edge * num_pts)
    num_pts_edge = num_pts - num_pts_unif

    my_data = np.genfromtxt(
        os.path.dirname(os.path.realpath(__file__)) + "/raw_dataset.csv", delimiter=","
    )
    x1 = my_data[0:, 0:6]
    y_raw = my_data[0:, 6:]
    x_raw = reshape_cos_sin(x1)  # pass input angles through sin cos filter

    y_mean = (
        np.sum(y_raw, 0) / np.sum(y_raw, 0).flatten()[-1]
    )  # the mean point of output
    y_dist = np.array(
        [
            np.linalg.norm(y_raw[i, 0:3] - y_mean.flatten()[0:3])
            for i in range(y_raw.shape[0])
        ]
    )
    y_dist = np.sort(y_dist, axis=None)
    y_radii = np.sum(y_dist[-10:]) / 10  # avg radius of output ball

    ## randomly select data samples inside the output ball
    random_indices = np.random.choice(x_raw.shape[0], size=num_pts_unif, replace=False)
    x = x_raw[random_indices, :]
    y = y_raw[random_indices, :]
    x_raw = np.delete(x_raw, random_indices, axis=0)
    y_raw = np.delete(y_raw, random_indices, axis=0)

    ## randomly select data samples on the edge of output ball
    counter = 0
    while counter != num_pts_edge:
        indx = np.random.randint(0, x_raw.shape[0])
        if (
            np.linalg.norm(y_raw[indx, 0:3] - y_mean.flatten()[0:3])
            >= edge_scale * y_radii
        ):
            x = np.vstack((x, x_raw[indx, :]))
            y = np.vstack((y, y_raw[indx, :]))
            x_raw = np.delete(x_raw, indx, axis=0)
            y_raw = np.delete(y_raw, indx, axis=0)
            counter += 1

    return x, y
