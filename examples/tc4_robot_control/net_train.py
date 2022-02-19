""" This script train a DNN for the car control example in https://arxiv.org/pdf/2109.14041.pdf
ref: https://arxiv.org/pdf/2109.14041.pdf
example: Car Control
network arch: 3-10-10-3

Returns:
    _type_: _description_
"""

import os
import pickle
import argparse
from tensorflow import keras
from rc_utils import CarControlProblem
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
        const=100,
        default=100,
        help="Specify training epochs in int, default: 5000",
    )
    parser.add_argument(
        "-lr",
        "--learnRate",
        nargs="?",
        type=float,
        const=0.003,
        default=0.003,
        help="Specify Learning rate in int, default: 0.003",
    )
    parser.add_argument(
        "-rr",
        "--regularizationRate",
        nargs="?",
        type=float,
        const=0.0001,
        default=0.0001,
        help="Specify regularization rate in int, default: 0.001",
    )
    parser.add_argument(
        "-bs",
        "--batchSizeTrain",
        nargs="?",
        type=int,
        const=50,
        default=50,
        help="Specify training batch sizes at each epoch in int, default: 50",
    )
    parser.add_argument(
        "-vi",
        "--visualization",
        nargs="?",
        type=int,
        const=1,
        default=1,
        choices=range(0, 2),
        help="Specify visualization variable 1 = on, 0 = off, default: 1",
    )
    return parser.parse_args()


def main(
    direc,
    learning_rate,
    regularizer_rate,
    train_epochs,
    visual,
    batch_size_train,
):
    """_summary_

    Args:
        direc (_type_): _description_
        learning_rate (_type_): _description_
        regularizer_rate (_type_): _description_
        train_epochs (_type_): _description_
        visual (_type_): _description_
        batch_size_train (_type_): _description_
    """
    path = direc + "/tc4/original_net"

    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")

    #########################################
    # parameters
    ## Data samples
    num_traj = 100  # number of samples
    train2test_ratio = 0.7
    ## Network
    input_dim = 4
    output_dim = 3
    hid_dim_0 = 10
    hid_dim_1 = 10

    #########################################
    # DNN arch definition
    print("-----------------------")
    print("NN model construction - model summery:")

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
    ## loss define
    loss = keras.losses.MeanSquaredError(name="MSE")
    ## optimizer define
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, name="Adam")
    ## compile the model
    model_orig.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    model_orig.summary()

    #########################################
    # Construct CarControlProblem Object
    control_obj = CarControlProblem(model_orig)
    control_obj.initialize_sym_states()

    ## Start model training
    train_set, test_set = control_obj.apply_dagger_learn(
        num_traj, train_epochs, batch_size_train, train2test_ratio
    )

    #########################################
    # Visualization
    if visual == 1:
        control_obj.plot_history()
    #########################################
    # saving model
    print("-----------------------")
    print("the data set and model are saved in {}".format(path))
    if not os.path.exists(path):
        os.makedirs(path + "/model")
    keras.models.save_model(
        control_obj.controller_nn,
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
    with open(path + "/data/input_output_data_tc4.pickle", "wb") as data:
        pickle.dump([train_set, test_set], data)


if __name__ == "__main__":
    args = arg_parser()
    main(
        args.path,
        args.learnRate,
        args.regularizationRate,
        args.epoch,
        args.visualization,
        args.batchSizeTrain,
    )
