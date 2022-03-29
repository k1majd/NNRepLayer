"""_summary_

Returns:
    _type_: _description_
"""
import os
import pickle
import numpy as np
import random
from tensorflow import keras


def original_model_loader():
    """_summary_

    Args:
        json_path (_type_): _description_
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    direc = os.path.dirname(os.path.realpath(__file__)) + "/raw_data"
    json_path = os.path.join(direc, "ACASXU_2_9.json")
    model_path = os.path.join(direc, "ACASXU_2_9.h5")
    # load json and create model
    json_file = open(json_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path)
    print("Loaded model from disk")
    return loaded_model


def in_bound_input(inp, inp_bounds):
    """_summary_

    Args:
        inp (_type_): _description_
        bounds (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_temp = inp.flatten()
    for idx, bound in enumerate(inp_bounds):
        if not bound[0] <= x_temp[idx] <= bound[1]:
            return False
    return True


def sample_spherical(npoints, ndim=5):
    """_summary_

    Args:
        npoints (_type_): _description_
        ndim (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def vacinity_adv_finder(
    num_pts, x_adv_prev, y_adv_prev, model, model_type, inp_bounds, res=7
):
    """_summary_

    Args:
        num_iters (_type_): _description_
        x_adv_prev (_type_): _description_
        y_adv_prev (_type_): _description_
        model (_type_): _description_
        model_type (_type_): _description_
        res (int, optional): _description_. Defaults to 7.

    Returns:
        _type_: _description_
    """
    x_center = np.array(x_adv_prev)
    x_adv_set = x_adv_prev.reshape((1, 5))
    y_adv_set = y_adv_prev.reshape((1, 5))
    precision = [10, 50, 100, 200, 500, 700, 1000]
    pts_so_far = 0
    while pts_so_far < num_pts:
        p = random.choice(precision)
        m = sample_spherical(1, ndim=5)
        in_temp = x_center + m.flatten() / p
        in_temp = in_temp.reshape((1, 5))
        if model_type == "tf":
            out_temp = model.predict(in_temp)
        else:
            _, _, _, _, _, _, out_temp = model(in_temp, relu=False)
        out_temp[0] = np.round(out_temp[0], res)
        if not (
            (
                out_temp[0][0] < out_temp[0][2]
                and out_temp[0][0] < out_temp[0][3]
                and out_temp[0][0] < out_temp[0][4]
            )
            or (
                out_temp[0][1] < out_temp[0][2]
                and out_temp[0][1] < out_temp[0][3]
                and out_temp[0][1] < out_temp[0][4]
            )
        ):
            if in_bound_input(in_temp, inp_bounds):
                x_adv_set = np.concatenate((x_adv_set, in_temp), axis=0)
                y_adv_set = np.concatenate((y_adv_set, out_temp), axis=0)
                print(f"point:{pts_so_far + 1}")
                print(f"the precision distance of found point: {p}")
                pts_so_far += 1
    return x_adv_set, y_adv_set


def train_test_sampler(model, x_adv, num_train, num_test):
    """_summary_

    Args:
        model (keras): original model
        x_adv (ndarray): adversarial point
        num_train (int): number of train samples - adv in training
        num_test (int): number of test samples - adv in testing
    """
    x_center = np.array(x_adv)
    precision = [100, 10, 5, 1, 0.05, 0.1]
    x_train = []
    y_train = []
    x_test = []
    y_test = []


# def original_data_loader():
#     """_summary_

#     Raises:
#         ImportError: error if the data is does not exist in the designated location

#     Returns:
#         list[ndarray]: a list of [x_train, y_train, x_test, y_test]
#     """
#     direc = os.path.dirname(os.path.realpath(__file__))
#     path_read = direc + "/tc5/original_net"
#     if not os.path.exists(path_read + "/data/input_output_data_tc5.pickle"):
#         raise ImportError(
#             "path {path_read}/data/input_output_data_tc5.pickle does not exist!"
#         )
#     with open(path_read + "/data/input_output_data_tc5.pickle", "rb") as data:
#         dataset = pickle.load(data)
#     return dataset[0], dataset[1], dataset[2], dataset[3]


# def wm_data_loader(model_orig):
#     """_summary_

#     Args:
#         model_orig (_type_): _description_

#     Raises:
#         ImportError: _description_

#     Returns:
#         _type_: _description_
#     """
#     direc = os.path.dirname(os.path.realpath(__file__))
#     path_read = direc + "/tc5/original_net"
#     if not os.path.exists(path_read + "/data/input_output_wm_tc5.pickle"):
#         raise ImportError(
#             "path {path_read}/data/input_output_wm_tc5.pickle does not exist!"
#         )
#     with open(path_read + "/data/input_output_wm_tc5.pickle", "rb") as data:
#         dataset = pickle.load(data)
#     return dataset[0], model_orig.predict(dataset[0]), dataset[1]
