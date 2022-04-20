import pickle
from ctypes import util
import unittest

import pickle
from nnreplayer.form_nn.dense import Dense
from nnreplayer.utils.utils import tf2_get_architecture
import filecmp
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os
from tensorflow import keras

# from nnreplayer.utils.utils import generate_outside_constraints

class TestDense(unittest.TestCase):
    def test1_init(self):
        n_in = 0
        n_out = 1

        # layer = Dense(n_in, n_out, False, 12345) 
        self.assertRaises(ValueError, Dense, n_in, n_out, False, 12345)
    
    def test2_init(self):
        n_in = 1
        n_out = 0

        # layer = Dense(n_in, n_out, False, 12345) 
        self.assertRaises(ValueError, Dense, n_in, n_out, False, 12345)

    def test3_init(self):
        n_in = 1
        n_out = 1

        layer = Dense(n_in, n_out, False, 12345) 
        
        with open("./tests/form_nn/dense_resources/gold_layer.pkl", "rb") as f:
            gold_layer = pickle.load(f)

        np.testing.assert_array_equal(layer.weights, gold_layer.weights)
        np.testing.assert_array_equal(layer.bias, gold_layer.bias)
        np.testing.assert_array_equal(layer.relu, gold_layer.relu)
    
    def test4_init(self):
        n_in = 300
        n_out = 300

        layer = Dense(n_in, n_out, False, 12345) 
        
        with open("./tests/form_nn/dense_resources/gold_layer_300.pkl", "rb") as f:
            gold_layer = pickle.load(f)
            

        np.testing.assert_array_equal(layer.weights, gold_layer.weights)
        np.testing.assert_array_equal(layer.bias, gold_layer.bias)
        np.testing.assert_array_equal(layer.relu, gold_layer.relu)

    def test1__relu(self):
        n_in = 300
        n_out = 300

        rng = np.random.default_rng(12345)
        x = rng.random((n_in, n_out))
        
        with open("./tests/form_nn/dense_resources/gold__relu_output.pkl", "rb") as f:
            gold_relu_output = pickle.load(f)
            # pickle.dump(x,f)
        
        layer = Dense(n_in, n_out, False, 12345)
        pred = layer._relu(x)
        true = layer._relu(gold_relu_output)
        np.testing.assert_array_equal(pred, true)

    
    def test1_call(self):
        input_features = 300
        n_in = 300
        n_out = 400
        rng = np.random.default_rng(12)
        layer = Dense(n_in, n_out, False, 12345)
        inputs = rng.random((1,input_features))

        x = layer(inputs)
       
        with open("./tests/form_nn/dense_resources/gold__call_relu_false_output.pkl", "rb") as f:
            # pickle.dump(x,f)
            gold_relu_false_output = pickle.load(f)
            
        
        
        np.testing.assert_array_equal(gold_relu_false_output, x)

    def test2_call(self):               
        input_features = 300
        n_in = 300
        n_out = 400
        rng = np.random.default_rng(12)
        layer = Dense(n_in, n_out, True, 12345)
        inputs = rng.random((1,input_features))

        x = layer(inputs)

        with open("./tests/form_nn/dense_resources/gold__call_relu_true_output.pkl", "rb") as f:
            gold_relu_true_output = pickle.load(f)

        np.testing.assert_array_equal(gold_relu_true_output, x)

    def test_set_variables(self):
        n_in = 300
        n_out = 400
        rng = np.random.default_rng(12)
        
        layer = Dense(n_in, n_out, False, 12356)
        new_weights = rng.random((n_in, n_out)) * 2 - 1
        new_bias = rng.random((n_out)) * 2 - 1

        layer.set_variables(new_weights, new_bias)
        
        np.testing.assert_array_equal(layer.weights, new_weights)
        np.testing.assert_array_equal(layer.bias, new_bias)

if __name__ == '__main__':
    unittest.main()