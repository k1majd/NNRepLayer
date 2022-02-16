# this script train a DNN for the car control example in https://arxiv.org/pdf/2109.14041.pdf
# ref: https://arxiv.org/pdf/2109.14041.pdf
# example: Car Control
# network arch: 3-10-10-3
#
import tensorflow as tf
from tensorflow import keras
import numpy as np
from fk_utils import data_generate, Batch
import pickle
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def arg_parser():
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
        const=5000,
        default=5000,
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
    args = parser.parse_args()
    return args


def main(
    direc,
    learning_rate,
    regularizer_rate,
    train_epochs,
    visual,
    batch_size_train,
):
    path = direc + "/tc2/original_net"

    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")

    #########################################
    # parameters
    ## Data samples
    num_pts = 600  # number of samples
    train2test_ratio = 0.7
    ## Network
    input_dim = 12
    output_dim = 4
    hid_dim_0 = 30
    hid_dim_1 = 30
    hid_dim_2 = 30

    #########################################
    # Generate dataset
    print("-----------------------")
    print("Data Generation")

    ## generate random samples
    x, y = data_generate(num_pts)

    ## construct a data batch class
    batch_size = int(train2test_ratio * num_pts)
    batch = Batch(x, y, batch_size)
    print("data size: {}".format(num_pts))
    print("train2test ratio: {}".format(train2test_ratio))

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
            hid_dim_2,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(regularizer_rate),
            bias_regularizer=keras.regularizers.l2(regularizer_rate),
            name="layer2",
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

    #########################################
    # compile the model
    print("-----------------------")
    print("Start training!")
    ## loss define
    loss = keras.losses.MeanSquaredError(name="MSE")
    ## optimizer define
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, name="Adam")
    ## compile the model
    model_orig.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    x_train, y_train, x_test, y_test = batch.getBatch()  # create data batch
    ## define callbacks:
    callback_reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
    )  # reduce learning rate
    callback_es = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )  # early stopping callback
    ## model fitting
    his = model_orig.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=train_epochs,
        batch_size=batch_size_train,
        use_multiprocessing=True,
        verbose=1,
        callbacks=[callback_es, callback_reduce_lr],
    )
    print("Model Loss + Accuracy on Train Data Set: ")
    model_orig.evaluate(x_train, y_train, verbose=2)  # model evaluation
    print("Model Loss + Accuracy on Test Data Set: ")
    model_orig.evaluate(x_test, y_test, verbose=2)  # model evaluation

    #########################################
    # visualization
    if visual == 1:
        print("----------------------")
        print("Visualization")
        plt.rcParams["text.usetex"] = True
        mpl.style.use("seaborn")

        ## loss plotting
        results_train_loss = his.history["loss"]
        results_valid_loss = his.history["val_loss"]
        plt.plot(results_train_loss, color="red", label="training loss")
        plt.plot(results_valid_loss, color="blue", label="validation loss")
        plt.title("Loss Function Output")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper left", frameon=False)
        plt.show()

        ## accuracy plotting
        results_train_acc = his.history["accuracy"]
        results_valid_acc = his.history["val_accuracy"]
        plt.plot(results_train_acc, color="red", label="training accuracy")
        plt.plot(results_valid_acc, color="blue", label="validation accuracy")
        plt.title("Accuracy Function Output")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="upper left", frameon=False)
        plt.show()

        ## predicted output (training dataset)
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            y_train[:, 0],
            y_train[:, 1],
            y_train[:, 2],
            color="tab:blue",
            label="Original Target",
        )
        y_predict_train = model_orig.predict(x_train)
        ax.scatter3D(
            y_predict_train[:, 0],
            y_predict_train[:, 1],
            y_predict_train[:, 2],
            color="mediumseagreen",
            label="Predicted Target",
        )
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        fig.tight_layout()

        # plt.title('Repaired Model')
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.zaxis.label.set_size(15)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="z", labelsize=15)
        plt.legend(loc="upper left", frameon=False, fontsize=20)
        plt.title(r"Forward Kinematics (training dataset)", fontsize=25)
        plt.show()

        ## predicted output (testing dataset)
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            y_test[:, 0],
            y_test[:, 1],
            y_test[:, 2],
            color="tab:blue",
            label="Original Target",
        )
        y_predict_test = model_orig.predict(x_test)
        ax.scatter3D(
            y_predict_test[:, 0],
            y_predict_test[:, 1],
            y_predict_test[:, 2],
            color="mediumseagreen",
            label="Predicted Target",
        )
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        fig.tight_layout()
        # plt.title('Repaired Model')
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.zaxis.label.set_size(15)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.tick_params(axis="z", labelsize=15)
        plt.legend(loc="upper left", frameon=False, fontsize=20)
        plt.title(r"Forward Kinematics (testing dataset)", fontsize=25)
        plt.show()

    #########################################
    # saving model
    print(model_orig.get_weights()[0])
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
    with open(path + "/data/input_output_data_tc2.pickle", "wb") as data:
        pickle.dump([x_train, y_train, x_test, y_test], data)


if __name__ == "__main__":
    args = arg_parser()
    direc = args.path
    learning_rate = args.learnRate
    regularizer_rate = args.regularizationRate
    train_epochs = args.epoch
    visual = args.visualization
    batch_size_train = args.batchSizeTrain
    main(
        direc,
        learning_rate,
        regularizer_rate,
        train_epochs,
        visual,
        batch_size_train,
    )
