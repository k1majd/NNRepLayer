import os
import argparse
from tensorflow import keras
import numpy as np
from shapely.geometry import Polygon
from affine_utils import label_output_inside
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt
import matplotlib as mpl


def arg_parser():
    """_summary_

    Returns:
        _type_: _description_
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        nargs="?",
        default=cwd,
        help="Specify a path to store the data",
    )
    parser.add_argument(
        "-ep",
        "--epoch",
        nargs="?",
        type=int,
        default=500,
        help="Specify training epochs in int, default: 100",
    )
    parser.add_argument(
        "-lr",
        "--learnRate",
        nargs="?",
        type=float,
        default=0.01,
        help="Specify Learning rate in int, default: 0.003",
    )
    parser.add_argument(
        "-rr",
        "--regularizationRate",
        nargs="?",
        type=float,
        default=0.0001,
        help="Specify regularization rate in int, default: 0.0001",
    )
    parser.add_argument(
        "-bs",
        "--batchSizeTrain",
        nargs="?",
        type=int,
        default=50,
        help="Specify training batch sizes at each epoch in int, default: 50",
    )
    parser.add_argument(
        "-vi",
        "--visualization",
        nargs="?",
        type=int,
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
    """
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
    if not os.path.exists(path_read + "/data/input_output_data_tc1.pickle"):
        raise ImportError(
            "path {path_read}/data/input_output_data_tc1.pickle does not exist!"
        )
    with open(path_read + "/data/input_output_data_tc1.pickle", "rb") as data:
        dataset = pickle.load(data)
    x_train_inside, y_train_inside = label_output_inside(
        poly_const, dataset[0], dataset[1], mode="finetune"
    )
    x_test_inside, y_test_inside = label_output_inside(
        poly_const, dataset[2], dataset[3], mode="retrain"
    )
    plt.show()
    print("-----------------------")
    print("NN model fine tuning:")

    if not os.path.exists(path_read + "/model"):
        raise ImportError("path {path_read}/model does not exist!")
    model_orig = keras.models.load_model(path_read + "/model")

    # substitute the output layer with a new layer and freeze the base model
    output_dim = model_orig.layers[-1].output.shape[1]
    model_orig.pop()
    for lnum, layer in enumerate(model_orig.layers):
        layer.trainable = False
    model_orig.add(
        keras.layers.Dense(
            output_dim,
            kernel_regularizer=keras.regularizers.l2(regularizer_rate),
            bias_regularizer=keras.regularizers.l2(regularizer_rate),
            name="output",
        )
    )
    for lnum, layer in enumerate(model_orig.layers):
        print(lnum, layer.name, layer.trainable, layer.dtype, layer.dtype_policy)
    model_orig.summary()

    print("-----------------------")
    print("Start fine-tuning the last layer!")

    loss = keras.losses.MeanSquaredError(name="MSE")
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, name="Adam")
    model_orig.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    # compile the model
    callback_reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=10, min_lr=0.0001
    )  # reduce learning rate
    callback_es = EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True
    )  # early stopping callback
    ## model fitting
    # his = model_orig.fit(
    #     x_train_inside,
    #     y_train_inside,
    #     validation_data=(x_test_inside, y_test_inside),
    #     epochs=train_epochs,
    #     batch_size=batch_size_train,
    #     use_multiprocessing=True,
    #     verbose=1,
    #     callbacks=[callback_es, callback_reduce_lr],
    # )
    his = model_orig.fit(
        x_train_inside,
        y_train_inside,
        epochs=train_epochs,
        batch_size=batch_size_train,
        use_multiprocessing=True,
        verbose=1,
        callbacks=[callback_es, callback_reduce_lr],
    )
    print("Model Loss + Accuracy on Test Data Set: ")
    model_orig.evaluate(x_test_inside, y_test_inside, verbose=2)

    if visual == 1:
        print("----------------------")
        print("Visualization")
        plt.rcParams["text.usetex"] = False
        mpl.style.use("seaborn")

        ## loss plotting
        results_train_loss = his.history["loss"]
        # results_valid_loss = his.history["val_loss"]
        plt.plot(results_train_loss, color="red", label="training loss")
        # plt.plot(results_valid_loss, color="blue", label="validation loss")
        plt.title("Loss Function Output (fine-tuning the last layer)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper left", frameon=False)
        # plt.savefig(path[0] + "_acc." + path[1], format="eps")
        plt.show()

        ## accuracy plotting
        results_train_acc = his.history["accuracy"]
        # results_valid_acc = his.history["val_accuracy"]
        plt.plot(results_train_acc, color="red", label="training accuracy")
        # plt.plot(results_valid_acc, color="blue", label="validation accuracy")
        plt.title("Accuracy Function Output (fine-tuning the last layer)")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="upper left", frameon=False)
        plt.show()

    print("-----------------------")
    print("Start fine-tuning the whole model!")
    for lnum, layer in enumerate(model_orig.layers):
        layer.trainable = True

    loss = keras.losses.MeanSquaredError(name="MSE")
    optimizer = keras.optimizers.SGD(learning_rate=0.0001, name="Adam")
    model_orig.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    # compile the model
    callback_reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=10, min_lr=0.0001
    )  # reduce learning rate
    callback_es = EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True
    )  # early stopping callback
    ## model fitting
    # his = model_orig.fit(
    #     x_train_inside,
    #     y_train_inside,
    #     validation_data=(x_test_inside, y_test_inside),
    #     epochs=500,
    #     batch_size=batch_size_train,
    #     use_multiprocessing=True,
    #     verbose=1,
    #     callbacks=[callback_es, callback_reduce_lr],
    # )
    his = model_orig.fit(
        x_train_inside,
        y_train_inside,
        epochs=50,
        batch_size=batch_size_train,
        use_multiprocessing=True,
        verbose=1,
        callbacks=[callback_es, callback_reduce_lr],
    )
    print("Model Loss + Accuracy on Test Data Set: ")
    model_orig.evaluate(x_test_inside, y_test_inside, verbose=2)

    if visual == 1:
        print("----------------------")
        print("Visualization")
        plt.rcParams["text.usetex"] = False
        mpl.style.use("seaborn")

        ## loss plotting
        results_train_loss = his.history["loss"]
        # results_valid_loss = his.history["val_loss"]
        plt.plot(results_train_loss, color="red", label="training loss")
        # plt.plot(results_valid_loss, color="blue", label="validation loss")
        plt.title("Loss Function Output")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper left", frameon=False)
        # plt.savefig(path[0] + "_acc." + path[1], format="eps")
        plt.show()

        ## accuracy plotting
        results_train_acc = his.history["accuracy"]
        # results_valid_acc = his.history["val_accuracy"]
        plt.plot(results_train_acc, color="red", label="training accuracy")
        # plt.plot(results_valid_acc, color="blue", label="validation accuracy")
        plt.title("Accuracy Function Output")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend(loc="upper left", frameon=False)
        plt.show()

        x_poly_const_bound, y_poly_const_bound = poly_const.exterior.xy
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
        plt.plot(
            x_poly_const_bound,
            y_poly_const_bound,
            color="tab:red",
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
            label="Constraint Set",
        )
        plt.scatter(
            y_train_inside[:, 0],
            y_train_inside[:, 1],
            color="tab:blue",
            label="Original Target",
        )
        y_predict_train = model_orig.predict(x_train_inside)
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
        plt.plot(
            x_poly_const_bound,
            y_poly_const_bound,
            color="tab:red",
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
            label="Target Set",
        )
        plt.scatter(
            y_test_inside[:, 0],
            y_test_inside[:, 1],
            color="tab:blue",
            label="Original Target",
        )
        y_predict_test = model_orig.predict(x_test_inside)
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
    print(f"the data set and model are saved in {path_write}")

    if not os.path.exists(path_write):
        os.makedirs(path_write + "/model")
    keras.models.save_model(
        model_orig,
        path_write + "/model",
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )

    if not os.path.exists(path_write + "/data"):
        os.makedirs(path_write + "/data")
    with open(path_write + "/data/input_output_data_tc1.pickle", "wb") as data:
        pickle.dump(
            [x_train_inside, y_train_inside, x_test_inside, y_test_inside], data
        )


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
