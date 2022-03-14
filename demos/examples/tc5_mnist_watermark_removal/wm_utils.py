"""_summary_

Returns:
    _type_: _description_
"""
import os
from tensorflow import keras


def load_model():
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
