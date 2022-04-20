import pickle
from ctypes import util
import unittest

import pickle
from nnreplayer.form_nn.mlp import MLP
from nnreplayer.utils.utils import tf2_get_architecture
import filecmp
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import os
from tensorflow import keras

# from nnreplayer.utils.utils import generate_outside_constraints

class TestMLP(unittest.TestCase):
    def test_init(self):
        model_orig = keras.models.load_model("./tests/form_nn/affine_transform_original_model")
        model_orig.compile()
        
        architecture = tf2_get_architecture(model_orig)
        model_mlp = MLP(
            architecture[0],
            architecture[-1],
            architecture[1:-1],
        )
        # print(type(model_mlp.layers[0].weights))
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            # pickle.dump(model_mlp, f)
        
        for j in range(len(gold_layers.layers)):
            np.testing.assert_array_equal(model_mlp.layers[j].weights, gold_layers.layers[j].weights)
            np.testing.assert_array_equal(model_mlp.layers[j].bias, gold_layers.layers[j].bias)
            np.testing.assert_array_equal(model_mlp.layers[j].relu, gold_layers.layers[j].relu)
        
    def test_call(self):
        model_orig = keras.models.load_model("./tests/form_nn/affine_transform_original_model")
        model_orig.compile()
        
        architecture = tf2_get_architecture(model_orig)
        model_mlp = MLP(
            architecture[0],
            architecture[-1],
            architecture[1:-1],
        )
        # print(type(model_mlp.layers[0].weights))
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            # pickle.dump(model_mlp, f)

        rng = np.random.default_rng(12345)
        inputs = rng.random((200, architecture[0]))
        gold = gold_layers(inputs)
        obt = model_mlp(inputs)
        
        for x,y in zip(gold, obt):
            np.testing.assert_array_equal(x, y)

    def test_set_mlp_params(self):
        
        # print(type(model_mlp.layers[0].weights))
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            # pickle.dump(model_mlp, f)
        # new_weights = []
        # new_bias = []
        mlp_weights = []
        rng = np.random.default_rng(123)
        for layer in gold_layers.layers:
            # new_weights.append(rng.random(layer.weights))
            # new_bias.append(rng.random(layer.bias))
            mlp_weights.append(rng.random(layer.weights.shape))
            mlp_weights.append(rng.random(layer.bias.shape))
        
        gold_layers.set_mlp_params(mlp_weights)

        iterate = 0
        for layer in gold_layers.layers:
            np.testing.assert_array_equal(layer.weights, mlp_weights[iterate])
            iterate += 1
            np.testing.assert_array_equal(layer.bias, mlp_weights[iterate])
            iterate += 1

    def test1_set_mlp_params_layer(self):
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            
        mlp_weights = []
        rng = np.random.default_rng(123)
        for layer in gold_layers.layers:
            # new_weights.append(rng.random(layer.weights))
            # new_bias.append(rng.random(layer.bias))
            mlp_weights.append(rng.random(layer.weights.shape))
            mlp_weights.append(rng.random(layer.bias.shape))

        layer_to_set = 1
        gold_layers.set_mlp_params_layer(mlp_weights[0:2], layer_to_set)
        
        np.testing.assert_array_equal(gold_layers.layers[0].weights, mlp_weights[0])
        np.testing.assert_array_equal(gold_layers.layers[0].bias, mlp_weights[1])
        np.testing.assert_array_equal(gold_layers.layers[1].weights, gold_layers.layers[1].weights)
        np.testing.assert_array_equal(gold_layers.layers[1].bias, gold_layers.layers[1].bias)

        np.testing.assert_array_equal(gold_layers.layers[2].weights, gold_layers.layers[2].weights)
        np.testing.assert_array_equal(gold_layers.layers[2].bias, gold_layers.layers[2].bias)

    def test2_set_mlp_params_layer(self):
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            
        mlp_weights = []
        rng = np.random.default_rng(123)
        for layer in gold_layers.layers:
            # new_weights.append(rng.random(layer.weights))
            # new_bias.append(rng.random(layer.bias))
            mlp_weights.append(rng.random(layer.weights.shape))
            mlp_weights.append(rng.random(layer.bias.shape))

        layer_to_set = 2
        gold_layers.set_mlp_params_layer(mlp_weights[2:4], layer_to_set)
        
        np.testing.assert_array_equal(gold_layers.layers[1].weights, mlp_weights[2])
        np.testing.assert_array_equal(gold_layers.layers[1].bias, mlp_weights[3])
        np.testing.assert_array_equal(gold_layers.layers[0].weights, gold_layers.layers[0].weights)
        np.testing.assert_array_equal(gold_layers.layers[0].bias, gold_layers.layers[0].bias)

        np.testing.assert_array_equal(gold_layers.layers[2].weights, gold_layers.layers[2].weights)
        np.testing.assert_array_equal(gold_layers.layers[2].bias, gold_layers.layers[2].bias)
    
    def test3_set_mlp_params_layer(self):
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            
        mlp_weights = []
        rng = np.random.default_rng(123)
        for layer in gold_layers.layers:
            # new_weights.append(rng.random(layer.weights))
            # new_bias.append(rng.random(layer.bias))
            mlp_weights.append(rng.random(layer.weights.shape))
            mlp_weights.append(rng.random(layer.bias.shape))

        layer_to_set = 3
        gold_layers.set_mlp_params_layer(mlp_weights[4:6], layer_to_set)
        
        np.testing.assert_array_equal(gold_layers.layers[2].weights, mlp_weights[4])
        np.testing.assert_array_equal(gold_layers.layers[2].bias, mlp_weights[5])
        np.testing.assert_array_equal(gold_layers.layers[0].weights, gold_layers.layers[0].weights)
        np.testing.assert_array_equal(gold_layers.layers[0].bias, gold_layers.layers[0].bias)

        np.testing.assert_array_equal(gold_layers.layers[1].weights, gold_layers.layers[1].weights)
        np.testing.assert_array_equal(gold_layers.layers[1].bias, gold_layers.layers[1].bias)
        
    def test4_set_mlp_params_layer(self):
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            
        mlp_weights = []
        rng = np.random.default_rng(123)
        for layer in gold_layers.layers:
            # new_weights.append(rng.random(layer.weights))
            # new_bias.append(rng.random(layer.bias))
            mlp_weights.append(rng.random(layer.weights.shape))
            mlp_weights.append(rng.random(layer.bias.shape))

        layer_to_set = 0
        self.assertRaises(ValueError, lambda:  gold_layers.set_mlp_params_layer(mlp_weights[2:4], layer_to_set))
        layer_to_set = 4
        self.assertRaises(ValueError, lambda:  gold_layers.set_mlp_params_layer(mlp_weights[2:4], layer_to_set))
        
    def test1_get_mlp_weights(self):
        
        # print(type(model_mlp.layers[0].weights))
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            # pickle.dump(model_mlp, f)
        new_weights = []
        new_bias = []
        mlp_weights = []
        rng = np.random.default_rng(123)
        for layer in gold_layers.layers:
            w = rng.random(layer.weights.shape)
            b = rng.random(layer.bias.shape)
            new_weights.append(w)
            new_bias.append(b)
            mlp_weights.append(w)
            mlp_weights.append(b)
        
        gold_layers.set_mlp_params(mlp_weights)
        obtained_weights = gold_layers.get_mlp_weights()
        for x,y in zip(new_weights, obtained_weights):
            np.testing.assert_array_equal(x,y)

        obtained_bias = gold_layers.get_mlp_biases()
        for x,y in zip(new_bias, obtained_bias):
            np.testing.assert_array_equal(x,y)
        
        obtained_params = gold_layers.get_mlp_params()
        for x,y in zip(mlp_weights, obtained_params):
            np.testing.assert_array_equal(x,y)
    
    def test1_get_params_layer(self):
        
        # print(type(model_mlp.layers[0].weights))
        with open("./tests/form_nn/mlp_resources/gold_layers.pkl", "rb") as f:
            gold_layers = pickle.load(f)
            # pickle.dump(model_mlp, f)
        new_weights = []
        new_bias = []
        mlp_weights = []
        rng = np.random.default_rng(123)
        for layer in gold_layers.layers:
            w = rng.random(layer.weights.shape)
            b = rng.random(layer.bias.shape)
            new_weights.append(w)
            new_bias.append(b)
            mlp_weights.append(w)
            mlp_weights.append(b)
        
        gold_layers.set_mlp_params(mlp_weights)
        ob_b_1 = gold_layers.get_mlp_bias_layer(1)
        ob_b_2 = gold_layers.get_mlp_bias_layer(2)
        ob_b_3 = gold_layers.get_mlp_bias_layer(3)
        ob_w_1 = gold_layers.get_mlp_weight_layer(1)
        ob_w_2 = gold_layers.get_mlp_weight_layer(2)
        ob_w_3 = gold_layers.get_mlp_weight_layer(3)
        ob_params = [ob_w_1, ob_b_1,
                        ob_w_2, ob_b_2,
                        ob_w_3, ob_b_3]

        for x, y in zip(ob_params, mlp_weights):
            np.testing.assert_array_equal(x,y)

        self.assertRaises(ValueError, lambda: gold_layers.get_mlp_bias_layer(0))
        self.assertRaises(ValueError, lambda: gold_layers.get_mlp_bias_layer(4))
        self.assertRaises(ValueError, lambda: gold_layers.get_mlp_weight_layer(0))
        self.assertRaises(ValueError, lambda: gold_layers.get_mlp_weight_layer(4))

        ob_wb_1 = gold_layers.get_mlp_params_layer(1)
        np.testing.assert_array_equal(ob_wb_1[0], new_weights[0])
        np.testing.assert_array_equal(ob_wb_1[1], new_bias[0])

        ob_wb_2 = gold_layers.get_mlp_params_layer(2)
        np.testing.assert_array_equal(ob_wb_2[0], new_weights[1])
        np.testing.assert_array_equal(ob_wb_2[1], new_bias[1])

        ob_wb_3 = gold_layers.get_mlp_params_layer(3)
        np.testing.assert_array_equal(ob_wb_3[0], new_weights[2])
        np.testing.assert_array_equal(ob_wb_3[1], new_bias[2])

        self.assertRaises(ValueError, lambda: gold_layers.get_mlp_params_layer(0))
        self.assertRaises(ValueError, lambda: gold_layers.get_mlp_params_layer(4))

if __name__ == '__main__':
    unittest.main()