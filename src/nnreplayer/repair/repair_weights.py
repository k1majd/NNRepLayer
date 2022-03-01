
from utils import mlp_get_weights, mlp_set_weights
from mlp import MLP
from mip_nn_model import MIPNNModel
import pyomo.environ as pyo
import numpy as np
from tensorflow import keras

def repair_weights(model_orig, layer_to_repair, architecture, x_train, y_train, cost_function_weights):
# Extract weights from original model
    model_orig_params = mlp_get_weights(model_orig)
    weights = [model_orig_params[iterate] for iterate in range(0,2*(len(architecture)-1),2)]
    bias = [model_orig_params[iterate] for iterate in range(1,2*(len(architecture)-1),2)]

    mlp_orig = MLP(architecture[0], architecture[-1], architecture[1:-1])
    mlp_orig = mlp_set_weights(mlp_orig, model_orig_params)
    layer_values_train = mlp_orig(x_train, relu=False)

    num_samples = layer_values_train[layer_to_repair-2].shape[0]

# Set up the optimizer and cost function
    mip_model_layer = MIPNNModel(layer_to_repair, architecture, weights, bias)
    y_ = mip_model_layer(layer_values_train[layer_to_repair-2], (num_samples, architecture[layer_to_repair-1]), w_b_bound_error = 1)
    model_lay = mip_model_layer.model

    cost_expr = cost_function_weights(y_, y_train, num_samples, architecture[-1]) 

    # minimize error bound
    dw_l = 'dw' + str(layer_to_repair)
    cost_expr += getattr(model_lay, dw_l)

# Set cost function and optimize/solve
    model_lay.obj = pyo.Objective(expr=cost_expr)
    pyo.TransformationFactory('gdp.bigm').apply_to(model_lay)
    opt = pyo.SolverFactory('gurobi',solver_io="python")

    opt.solve(model_lay, tee=True)

# Initialize new weights and bias 
    new_weight = np.zeros((architecture[layer_to_repair-1], architecture[layer_to_repair]))
    new_bias = np.zeros((1, architecture[layer_to_repair]))
    
    for j in range(architecture[layer_to_repair]):
        new_bias[0, j] = eval("model_lay.b{}[j].value".format(layer_to_repair))
        for i in range(architecture[layer_to_repair-1]):
            
            new_weight[i, j] = eval("model_lay.w{}[i,j].value".format(layer_to_repair))

# Set new weights and bias
    model_new_params = []
    iterate = 0
    for j in range(len(architecture)-1):
        if j + 1 != layer_to_repair:
            model_new_params.append(model_orig_params[iterate])
            # print(model_orig_params[iterate].shape)
            iterate = iterate + 1
            model_new_params.append(model_orig_params[iterate])
            # print(model_orig_params[iterate].shape)
            iterate = iterate + 1
        else:
            # print(iterate)
            model_new_params.append(new_weight)
            # print(new_weight.shape)
            
            model_new_params.append(np.squeeze(new_bias))
            iterate = iterate + 2
        

# Define new model
    new_model = keras.models.clone_model(model_orig)
    weights_bias_iterate = 0
    for iterate in range(len(architecture)-1):
        new_model.layers[iterate].set_weights(model_new_params[weights_bias_iterate:weights_bias_iterate+2])
        weights_bias_iterate = weights_bias_iterate + 2

    return model_new_params, new_model