import pickle
from rc_utils import CarControlProblem
from tensorflow import keras
import os

path = os.getcwd() + "/examples/tc4_robot_control/tc4/original_net"

with open(path + "/data/input_output_data_tc4.pickle", "rb") as data:
    dataset = pickle.load(data)

orig_model = keras.models.load_model(path + "/model")
control_obj = CarControlProblem(orig_model)
control_obj.initialize_sym_states()
control_obj.visualize_ref_vs_nn(dataset[1])
