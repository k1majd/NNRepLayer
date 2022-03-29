import os
import numpy as np
from tensorflow import keras
from keras import backend as K
import argparse
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
from acas_utils import original_model_loader, vacinity_adv_finder
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
    path_write = direc + "/tc3/original_net"
    if not os.path.exists(path_write):
        os.makedirs(path_write)
        print("The new directory is created!")

    #########################################
    # Load raw model and store the original model
    model_orig = original_model_loader()
    model_orig.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    #########################################
    # Load raw data
    inp_bounds = [
        (-0.3284228772, 0.6798577687),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, 0.5),
    ]
    for i in [1, 2, 3]:
        globals()[f"x_adv{i}"] = np.genfromtxt(
            path_read + f"/ACASXU_2_9_input_adv{i}.csv", delimiter=","
        )
        globals()[f"y_adv{i}"] = np.genfromtxt(
            path_read + f"/ACASXU_2_9_output_adv{i}.csv", delimiter=","
        )

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(
        x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
    )
    x_test = x_test.reshape(
        x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
    )
    temp_idx = np.random.choice(x_train[0].shape[0], 750)
    x_train = x_train[temp_idx]
    y_train = y_train[temp_idx]
    temp_idx = np.random.choice(x_test[0].shape[0], 250)
    x_test = x_test[temp_idx]
    y_test = y_test[temp_idx]
    wm_images = np.load(path_read + "/wm_imgs.npy")  # watermark images
    wm_labels = np.loadtxt(
        path_read + "/wm_labels.txt", dtype="int32"
    )  # watermark labels
    wm_images = wm_images.reshape(
        wm_images.shape[0], wm_images.shape[1], wm_images.shape[2], 1
    )

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
    inp = model_orig.input  # input placeholder
    outputs = [
        layer.output for layer in model_orig.layers
    ]  # all layer outputs
    functors = [K.function([inp], out) for out in outputs]

    layer_outs_train = [func([x_train]) for func in functors]
    layer_outs_test = [func([x_test]) for func in functors]
    if not os.path.exists(path_write + "/data"):
        os.makedirs(path_write + "/data")
    with open(path_write + "/data/input_output_data_tc5.pickle", "wb") as data:
        pickle.dump(
            [
                layer_outs_train[1],
                layer_outs_train[-1],
                layer_outs_test[1],
                layer_outs_test[-1],
            ],
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
