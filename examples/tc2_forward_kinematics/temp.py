import tensorflow as tf
from tensorflow import keras
import numpy as np
from shapely.geometry import Polygon
from fk_utils import data_generate, Batch
import pickle
import os
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl

model_orig = keras.models.load_model(os.getcwd()+'/examples/tc2_forward_kinematics/tc2/original_net/model')
with open(os.getcwd()+"/examples/tc2_forward_kinematics/tc2/original_net/data/input_output_data_tc2.pickle", "rb") as file:
    loaded_data = pickle.load(file)

x_train= loaded_data[0]
y_train= loaded_data[1]
x_test = loaded_data[2]
y_test = loaded_data[3]
print(model_orig.get_weights()[0])
loss = keras.losses.MeanSquaredError(name='MSE')
optimizer = keras.optimizers.SGD(learning_rate = 0.0004, name='Adam')
model_orig.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
his = model_orig.fit(x_train, y_train, epochs = 10000, batch_size = 50, use_multiprocessing = True, verbose = 0)
print("Model Loss + Accuracy on Test Data Set: ")
model_orig.evaluate(x_test,  y_test, verbose=2)
print(model_orig.get_weights()[0])
keras.models.save_model(model_orig, os.getcwd()+'/examples/tc2_forward_kinematics/tc2/original_net/model', overwrite=True, include_optimizer=True, save_format=None, signatures=None, options=None, save_traces=True)

print('Hi')