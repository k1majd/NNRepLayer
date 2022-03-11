import os
import argparse
import pickle
from csv import writer
from tensorflow import keras
import numpy as np
from shapely.geometry import Polygon
from fk_utils import (
    label_output_inside,
    plot_history,
    plot_dataset,
    model_eval,
    original_data_loader,
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt


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
        default=0.003,
        help="Specify regularization rate in int, default: 0.0001",
    )
    parser.add_argument(
        "-bs",
        "--batchSizeTrain",
        nargs="?",
        type=int,
        default=10,
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
    parser.add_argument(
        "-sm",
        "--saveModel",
        nargs="?",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Specify whether to save model or not 1 = on, 0 = off, default: 1",
    )
    parser.add_argument(
        "-sd",
        "--saveData",
        nargs="?",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Specify whether to save data or not 1 = on, 0 = off, default: 1",
    )
    parser.add_argument(
        "-ss",
        "--saveStats",
        nargs="?",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Specify whether to save stats or not 1 = on, 0 = off, default: 1",
    )
    parser.add_argument(
        "-vb",
        "--netVerbose",
        nargs="?",
        type=int,
        default=1,
        choices=range(0, 2),
        help="Specify whether to show training verbose or not 1 = on, 0 = off, default: 1",
    )
    return parser.parse_args()


def main(
    direc,
    learning_rate,
    regularizer_rate,
    train_epochs,
    visual,
    batch_size_train,
    save_model,
    save_data,
    save_stats,
    net_verbose,
):
    """_summary_

    Args:
        direc (_type_): _description_
        learning_rate (_type_): _description_
        regularizer_rate (_type_): _description_
        train_epochs (_type_): _description_
        visual (_type_): _description_
    """
    path_read = direc + "/tc2/original_net"
    path_write = direc + "/tc2/finetuned_net"

    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print("Directory: {path_write} is created!")

    print("-----------------------")
    print("Data modification")
    x_train, y_train, x_test, y_test = original_data_loader()
    x_train_inside, y_train_inside = label_output_inside(
        poly_const, x_train, y_train, bound_error=0.25, mode="finetune"
    )
    print(f"fine-tuning size: {y_train_inside.shape[0]}")
    x_test_inside, y_test_inside = label_output_inside(
        poly_const, x_test, y_test, bound_error=0.25, mode="retrain"
    )
    plt.show()
    print("-----------------------")
    print("NN model fine tuning:")

    if not os.path.exists(path_read + "/model"):
        raise ImportError(f"path {path_read}/model does not exist!")
    model_orig = keras.models.load_model(path_read + "/model")

    # substitute the output layer with a new layer and freeze the base model
    output_dim = model_orig.layers[-1].output.shape[1]
    model_orig.pop()
    for _, layer in enumerate(model_orig.layers):
        layer.trainable = False
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
    print("Start fine-tuning the last layer!")

    loss = keras.losses.MeanSquaredError(name="MSE")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, name="Adam")
    model_orig.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    # compile the model
    callback_reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=10, min_lr=0.001
    )  # reduce learning rate
    callback_es = EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )  # early stopping callback
    ## model fitting
    his = model_orig.fit(
        x_train_inside,
        y_train_inside,
        epochs=train_epochs,
        batch_size=batch_size_train,
        use_multiprocessing=True,
        verbose=net_verbose,
        callbacks=[callback_es, callback_reduce_lr],
    )
    print("Model Loss + Accuracy on Test Data Set: ")
    model_orig.evaluate(x_test_inside, y_test_inside, verbose=2)

    if visual == 1:
        plot_history(his, include_validation=False)
        plot_dataset(
            [poly_orig, poly_trans, poly_const],
            [y_train_inside, model_orig.predict(x_train_inside)],
            label="training",
        )
        plot_dataset(
            [poly_orig, poly_trans, poly_const],
            [y_test_inside, model_orig.predict(x_test_inside)],
            label="testing",
        )

    print("-----------------------")
    print("Start fine-tuning the whole model!")
    for _, layer in enumerate(model_orig.layers):
        layer.trainable = True

    loss = keras.losses.MeanSquaredError(name="MSE")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, name="Adam")
    model_orig.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    # compile the model
    callback_reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.2, patience=5, min_lr=0.001
    )  # reduce learning rate
    callback_es = EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True
    )  # early stopping callback
    ## model fitting
    his = model_orig.fit(
        x_train_inside,
        y_train_inside,
        epochs=50,
        batch_size=batch_size_train,
        use_multiprocessing=True,
        verbose=net_verbose,
        callbacks=[callback_es, callback_reduce_lr],
    )
    print("Model Loss + Accuracy on Test Data Set: ")
    model_orig.evaluate(x_test_inside, y_test_inside, verbose=2)

    if visual == 1:
        plot_history(his, include_validation=False)
        plot_dataset(
            [poly_orig, poly_trans, poly_const],
            [y_train_inside, model_orig.predict(x_train_inside)],
            label="training",
        )
        plot_dataset(
            [poly_orig, poly_trans, poly_const],
            [y_test_inside, model_orig.predict(x_test_inside)],
            label="testing",
        )

    print("-----------------------")
    print(f"the data set and model are saved in {path_write}")

    if save_model == 1:
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
        print("saved: model")

    if save_data == 1:
        if not os.path.exists(path_write + "/data"):
            os.makedirs(path_write + "/data")
        with open(
            path_write + "/data/input_output_data_tc2.pickle", "wb"
        ) as data:
            pickle.dump(
                [x_train_inside, y_train_inside, x_test_inside, y_test_inside],
                data,
            )
        print("saved: dataset")

    # save the statistics
    if save_stats == 1:
        if not os.path.exists(path_write + "/stats"):
            os.makedirs(path_write + "/stats")

        with open(
            path_write + "/stats/fine_tune_accs_stats_tc2.csv",
            "a+",
            newline="",
        ) as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)

            # Add contents of list as last row in the csv file
            csv_writer.writerow(
                model_eval(
                    model_orig,
                    keras.models.load_model(path_read + "/model"),
                    path_read,
                    poly_const,
                )
            )
        print("saved: stats")


if __name__ == "__main__":
    args = arg_parser()
    main(
        args.path,
        args.learnRate,
        args.regularizationRate,
        args.epoch,
        args.visualization,
        args.batchSizeTrain,
        args.saveModel,
        args.saveData,
        args.saveStats,
        args.netVerbose,
    )
