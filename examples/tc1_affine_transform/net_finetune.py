import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from shapely.geometry import Polygon, Point
from affine_utils import gen_rand_points_within_poly, Batch, label_output_inside
import pickle
from matplotlib import pyplot as plt
import matplotlib as mpl


def arg_parser():
    cwd = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        nargs="?",
        const=cwd,
        default=cwd,
        help="Specify a path to store the data",
    )
    parser.add_argument(
        "-ep",
        "--epoch",
        nargs="?",
        type=int,
        const=1000,
        default=1000,
        help="Specify training epochs in int",
    )
    parser.add_argument(
        "-lr",
        "--learnRate",
        nargs="?",
        type=int,
        const=0.003,
        default=0.003,
        help="Specify Learning rate in int",
    )
    parser.add_argument(
        "-rr",
        "--regularizationRate",
        nargs="?",
        type=int,
        const=0.001,
        default=0.001,
        help="Specify regularization rate in int",
    )
    parser.add_argument(
        "-vi",
        "--visualization",
        nargs="?",
        type=int,
        const=1,
        default=1,
        choices=range(0, 2),
        help="Specify visualization variable 1 = on, 0 = off",
    )
    args = parser.parse_args()
    return args


def main(direc, learning_rate, regularizer_rate, train_epochs, visual):
    path_read = direc + "/tc1/original_net"
    path_write = direc + "/tc1/finetuned_net"

    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print("Directory: {path_write} is created!")

    ## affine transformation matrices
    translate1 = np.array([[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]])  # translation matrix 1
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
    poly_trans = Polygon([(2.5, 4.621), (4.624, 2.5), (2.5, 0.3787), (0.3787, 2.5)])
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

    print("-----------------------")
    print("Data modification")
    with open(path_read + "/data/input_output_data_tc1.pickle", "rb") as data:
        dataset = pickle.load(data)
    y_train_inside = label_output_inside(poly_const, dataset[1])
    y_test_inside = label_output_inside(poly_const, dataset[3])
    print("I'm here")


if __name__ == "__main__":
    args = arg_parser()
    direc = args.path
    learning_rate = args.learnRate
    regularizer_rate = args.regularizationRate
    train_epochs = args.epoch
    visual = args.visualization
    main(direc, learning_rate, regularizer_rate, train_epochs, visual)
