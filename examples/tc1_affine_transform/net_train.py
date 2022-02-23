# this script train a DNN for the affine transformation example in https://arxiv.org/pdf/2109.14041.pdf
# ref: https://arxiv.org/pdf/2109.14041.pdf
# example: In-place rotation
# network arch: 3-10-10-3
#
import tensorflow as tf
from tensorflow import keras
import numpy as np
from shapely.geometry import Polygon
from affine_utils import gen_rand_points_within_poly, get_batch
import pickle
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl


def arg_parser():
    """_summary_

    Returns:
        _type_: _description_
    """
    cwd = os.getcwd()
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
    """_summary_

    Args:
        direc (_type_): _description_
        learning_rate (_type_): _description_
        regularizer_rate (_type_): _description_
        train_epochs (_type_): _description_
        visual (_type_): _description_
    """
    path = direc + "/tc1/original_net"

    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")

    # parameters
    ## Data samples
    num_pts = 300  # number of samples
    train2test_ratio = 0.7
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
    vert_const_inp = np.array(
        [[1.25, 3.75, 3.75, 1.25], [1.25, 1.25, 3.75, 3.75], [1, 1, 1, 1]]
    )  # contraint vertices in input space
    vert_const_out = np.matmul(
        np.matmul(np.matmul(translate1, rotate), translate2), vert_const_inp
    )  # constraint vertices in output space
    poly_const = Polygon(
        [
            (vert_const_out[0, 0], vert_const_out[1, 0]),
            (vert_const_out[0, 1], vert_const_out[1, 1]),
            (vert_const_out[0, 2], vert_const_out[1, 2]),
            (vert_const_out[0, 3], vert_const_out[1, 3]),
        ]
    )
    ## Network
    input_dim = 3
    output_dim = 3
    hid_dim_0 = 10
    hid_dim_1 = 10

    print("-----------------------")
    print("Data Generation")

    # Generate dataset
    ## generate random samples
    x = gen_rand_points_within_poly(poly_orig, num_pts)
    y = np.matmul(np.matmul(np.matmul(translate1, rotate), translate2), x.T)
    y = y.T
    ## construct a data batch class
    batch_size = int(train2test_ratio * num_pts)
    batch = Batch(x, y, batch_size)
    print("data size: {}".format(num_pts))
    print("train2test ratio: {}".format(train2test_ratio))

    print("-----------------------")
    print("NN model construction - model summery:")

    # DNN arch definition
    ## Define Sequential model with 3 layers
    model_orig = keras.Sequential(name="3_layer_NN")
    model_orig.add(
        keras.layers.Dense(
            hid_dim_0,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(regularizer_rate),
            bias_regularizer=keras.regularizers.l2(regularizer_rate),
            input_shape=(input_dim,),
            name="layer0",
        )
    )
    model_orig.add(
        keras.layers.Dense(
            hid_dim_1,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(regularizer_rate),
            bias_regularizer=keras.regularizers.l2(regularizer_rate),
            name="layer1",
        )
    )
    model_orig.add(
        keras.layers.Dense(
            output_dim,
            kernel_regularizer=keras.regularizers.l2(regularizer_rate),
            bias_regularizer=keras.regularizers.l2(regularizer_rate),
            name="output",
        )
    )

    model_orig.summary()

    print("-----------------------")
    print("Start training!")
    # compile the model
    loss = keras.losses.MeanSquaredError(name="MSE")
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, name="Adam")
    model_orig.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    x_train, y_train, x_test, y_test = batch.get_batch()
    his = model_orig.fit(
        x_train, y_train, epochs=train_epochs, use_multiprocessing=True, verbose=0
    )
    print("Model Loss + Accuracy on Test Data Set: ")
    model_orig.evaluate(x_test, y_test, verbose=2)

    if visual == 1:
        print("----------------------")
        print("Visualization")
        plt.rcParams["text.usetex"] = True
        mpl.style.use("seaborn")

        x_poly_trans_bound, y_poly_trans_bound = poly_trans.exterior.xy
        x_poly_orig_bound, y_poly_orig_bound = poly_orig.exterior.xy

        ## predicted output (training dataset)
        plt.plot(
            x_poly_orig_bound,
            y_poly_orig_bound,
            color="plum",
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
            label="Original Set",
        )
        plt.plot(
            x_poly_trans_bound,
            y_poly_trans_bound,
            color="tab:blue",
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
            label="Target Set",
        )
        plt.scatter(
            y_train[:, 0], y_train[:, 1], color="tab:blue", label="Original Target"
        )
        y_predict_train = model_orig.predict(x_train)
        plt.scatter(
            y_predict_train[:, 0],
            y_predict_train[:, 1],
            color="mediumseagreen",
            label="Predicted Target",
        )
        plt.legend(loc="upper left", frameon=False, fontsize=20)
        plt.title(r"In-place Rotation (training dataset)", fontsize=25)
        plt.xlabel("x", fontsize=25)
        plt.ylabel("y", fontsize=25)
        plt.show()

        ## predicted output (testing dataset)
        plt.plot(
            x_poly_orig_bound,
            y_poly_orig_bound,
            color="plum",
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
            label="Original Set",
        )
        plt.plot(
            x_poly_trans_bound,
            y_poly_trans_bound,
            color="tab:blue",
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
            label="Target Set",
        )
        plt.scatter(
            y_test[:, 0], y_test[:, 1], color="tab:blue", label="Original Target"
        )
        y_predict_test = model_orig.predict(x_test)
        plt.scatter(
            y_predict_test[:, 0],
            y_predict_test[:, 1],
            color="mediumseagreen",
            label="Predicted Target",
        )
        plt.legend(loc="upper left", frameon=False, fontsize=20)
        plt.title(r"In-place Rotation (testing dataset)", fontsize=25)
        plt.xlabel("x", fontsize=25)
        plt.ylabel("y", fontsize=25)
        plt.show()

    print("-----------------------")
    print("the data set and model are saved in {}".format(path))
    if not os.path.exists(path):
        os.makedirs(path + "/model")
    keras.models.save_model(
        model_orig,
        path + "/model",
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )

    if not os.path.exists(path + "/data"):
        os.makedirs(path + "/data")
    with open(path + "/data/input_output_data_tc1.pickle", "wb") as data:
        pickle.dump([x_train, y_train, x_test, y_test], data)


if __name__ == "__main__":
    args = arg_parser()
    direc = args.path
    learning_rate = args.learnRate
    regularizer_rate = args.regularizationRate
    train_epochs = args.epoch
    visual = args.visualization
    main(direc, learning_rate, regularizer_rate, train_epochs, visual)
