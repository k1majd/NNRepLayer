"""_summary_

Returns:
    _type_: _description_
"""
import os
import pickle
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
    json_path = os.path.join(direc, "mnist_original_model.json")
    model_path = os.path.join(direc, "mnist_original_model.h5")
    # load json and create model
    json_file = open(json_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path)
    print("Loaded model from disk")
    return loaded_model


def original_data_loader():
    """_summary_

    Raises:
        ImportError: error if the data is does not exist in the designated location

    Returns:
        list[ndarray]: a list of [x_train, y_train, x_test, y_test]
    """
    direc = os.path.dirname(os.path.realpath(__file__))
    path_read = direc + "/tc5/original_net"
    if not os.path.exists(path_read + "/data/input_output_data_tc5.pickle"):
        raise ImportError(
            "path {path_read}/data/input_output_data_tc5.pickle does not exist!"
        )
    with open(path_read + "/data/input_output_data_tc5.pickle", "rb") as data:
        dataset = pickle.load(data)
    return dataset[0], dataset[1], dataset[2], dataset[3]


def wm_data_loader(model_orig):
    """_summary_

    Args:
        model_orig (_type_): _description_

    Raises:
        ImportError: _description_

    Returns:
        _type_: _description_
    """
    direc = os.path.dirname(os.path.realpath(__file__))
    path_read = direc + "/tc5/original_net"
    if not os.path.exists(path_read + "/data/input_output_wm_tc5.pickle"):
        raise ImportError(
            "path {path_read}/data/input_output_wm_tc5.pickle does not exist!"
        )
    with open(path_read + "/data/input_output_wm_tc5.pickle", "rb") as data:
        dataset = pickle.load(data)
    return dataset[0], model_orig.predict(dataset[0]), dataset[1]
