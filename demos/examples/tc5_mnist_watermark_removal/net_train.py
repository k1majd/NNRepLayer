import os
import numpy as np
from tensorflow import keras
from keras import backend as K
import argparse
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from wm_utils import load_model
from nnreplayer.utils.utils import tf2_get_architecture


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
        const=cwd,
        default=cwd,
        help="Specify a path to store the data",
    )
    args = parser.parse_args()
    return args


def main(direc):
    """_summary_
    Args:
        direc (_type_): _description_
        learning_rate (_type_): _description_
        regularizer_rate (_type_): _description_
        train_epochs (_type_): _description_
        visual (_type_): _description_
        batch_size_train (_type_): _description_
    """
    #########################################
    # Path to store data
    path_read = direc + "/raw_data"
    path_write = direc + "/tc5/original_net"
    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print("The new directory is created!")

    #########################################
    # Load raw data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
    )
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
    )
    wm_images = np.load(path_read + "/wm_imgs.npy")  # watermark images
    wm_labels = np.loadtxt(
        path_read + "/wm_labels.txt", dtype="int32"
    )  # watermark labels
    wm_images = wm_images.reshape(
        wm_images.shape[0], wm_images.shape[1], wm_images.shape[2], 1
    )

    #########################################
    # Load raw model and store the original model
    net_model = load_model()
    net_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, name="Adam"),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    net_weights = net_model.get_weights()

    model_orig = keras.Sequential(name="3_layer_NN")
    model_orig.add(
        keras.layers.Dense(
            net_weights[0].shape[1],
            activation="relu",
            input_shape=(net_weights[0].shape[0],),
            name="layer0",
        )
    )
    model_orig.add(
        keras.layers.Dense(
            net_weights[-1].shape[1],
            name="output",
        )
    )
    # set the weights of replicated model
    model_orig.layers[0].set_weights([net_weights[0], net_weights[1]])
    model_orig.layers[1].set_weights([net_weights[2], np.zeros(10)])
    model_orig.summary()
    # save model:
    model_orig.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
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
    # test_loss, test_acc = net_model.evaluate(x_test, y_test)
    # train_loss, train_acc = net_model.evaluate(x_train, y_train)
    # wm_loss, wm_acc = net_model.evaluate(wm_images, wm_labels)

    #########################################
    # store data
    inp = net_model.input  # input placeholder
    outputs = [layer.output for layer in net_model.layers]  # all layer outputs
    functors = [K.function([inp], out) for out in outputs]

    layer_outs_train = [func([x_train]) for func in functors]
    layer_outs_test = [func([x_test]) for func in functors]
    if not os.path.exists(path_write + "/data"):
        os.makedirs(path_write + "/data")
    with open(path_write + "/data/input_output_data_tc5.pickle", "wb") as data:
        pickle.dump(
            [layer_outs_train[1], y_train, layer_outs_test[1], y_test],
            data,
        )
    print("saved: dataset - train, test")

    layer_outs_wm = [func([wm_images]) for func in functors]

    with open(path_write + "/data/input_output_wm_tc5.pickle", "wb") as data:
        pickle.dump(
            [layer_outs_wm[1], wm_labels],
            data,
        )
    print("saved: dataset - wm")


if __name__ == "__main__":
    args = arg_parser()
    direc = args.path
    main(direc)
