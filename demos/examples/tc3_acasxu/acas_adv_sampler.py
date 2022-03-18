import os
import numpy as np
from tensorflow import keras
import argparse
import pickle
from tensorflow import keras
import numpy as np
from acas_utils import original_model_loader, vacinity_adv_finder


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
    if not os.path.exists(path_write + "/data"):
        os.makedirs(path_write + "/data")
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

    adv_set_1 = vacinity_adv_finder(
        50, x_adv1, y_adv1, model_orig, "tf", inp_bounds, res=7
    )
    with open(
        path_write + "/data/input_output_adv1set_tc3.pickle", "wb"
    ) as data:
        pickle.dump(
            [adv_set_1[0], adv_set_1[1]],
            data,
        )
    print("saved: adv1 set")

    adv_set_2 = vacinity_adv_finder(
        50, x_adv2, y_adv2, model_orig, "tf", inp_bounds, res=7
    )
    with open(
        path_write + "/data/input_output_adv2set_tc3.pickle", "wb"
    ) as data:
        pickle.dump(
            [adv_set_2[0], adv_set_2[1]],
            data,
        )
    print("saved: adv2 set")

    adv_set_3 = vacinity_adv_finder(
        50, x_adv3, y_adv3, model_orig, "tf", inp_bounds, res=7
    )
    with open(
        path_write + "/data/input_output_adv3set_tc3.pickle", "wb"
    ) as data:
        pickle.dump(
            [adv_set_3[0], adv_set_3[1]],
            data,
        )
    print("saved: adv3 set")


if __name__ == "__main__":
    args = arg_parser()
    direc = args.path
    main(direc)
