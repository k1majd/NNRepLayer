import pyomo.environ as pyo
from pyomo.gdp import *
from .mip_layer import MIPLayer

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
        
        
    def __call__(self, x, shape, output_constraint_list, relu=False, weightSlack = 10, output_bounds=(-1e1, 1e1)):
        
        m, n = shape
        assert n == self.uin
        
        for layer in self.layers[:-1]:
            x = layer(x, (m, layer.uin), output_constraint_list, relu=True, weightSlack = weightSlack, output_bounds=output_bounds)
        
        layer = self.layers[-1]
        y = layer(x, (m, layer.uin), output_constraint_list, relu=relu, weightSlack = weightSlack, output_bounds=output_bounds)
        return y