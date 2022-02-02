import numpy as np
import os
import sys
from pprint import pprint
from numpy import sin, cos, pi
import numpy.matlib
import random
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, Point
import shapely.affinity as affinity
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import pyomo.environ as pyo
import pyomo.gdp as pyg
from pyomo.gdp import *
from mip_layer import MIPLayer

class MIPNNModel:
    def __init__(self, layer_to_repair, architecture, weights, bias, param_bounds=(-1, 1)):
        self.model = pyo.ConcreteModel()
        
        self.model.nlayers = layer_to_repair
        
        self.uin, self.uout = architecture[layer_to_repair-1], architecture[-1]
        uhidden = architecture[layer_to_repair:-1]
        
        self.layers = []
        prev = architecture[layer_to_repair-1]
        # print("UHidden = {}".format(uhidden))
        for iterate, u in enumerate(uhidden): 
            self.layers.append(MIPLayer(self.model, layer_to_repair, prev, u, weights[layer_to_repair-1 + iterate], bias[layer_to_repair-1 + iterate], param_bounds))
            prev = u
        self.layers.append(MIPLayer(self.model, layer_to_repair, prev, architecture[-1], weights[-1], bias[-1], param_bounds))
        
        
    def __call__(self, x, shape, A, b, relu=False, output_bounds=(-1e1, 1e1)):
        
        m, n = shape
        assert n == self.uin
        
        for layer in self.layers[:-1]:
            x = layer(x, (m, layer.uin), A,b, relu=True, output_bounds=output_bounds)
        
        layer = self.layers[-1]
        y = layer(x, (m, layer.uin), A,b, relu=relu, output_bounds=output_bounds)
        return y