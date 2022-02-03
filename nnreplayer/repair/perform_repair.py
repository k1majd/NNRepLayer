
from ..utils.results import Results
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow import keras
import tensorflow.keras.backend as kb

from .repair_weights_class import repair_weights

def perform_repair(layer_to_repair, model_orig, architecture, A,b,cost_function, train_dataset, options):
    x_train, y_train = train_dataset
    gdp_formulation = options.gdp_formulation
    solver_factory = options.solver_factory
    solver_language = options.solver_language
    model_output_type = options.model_output_type

    rep_weights = repair_weights(model_orig, layer_to_repair, architecture, A,b, cost_function)
    layer_values_train = rep_weights.extract_network(x_train)

    cost_expr, model_lay = rep_weights.set_up_optimizer(y_train, layer_values_train, weightSlack=options.weightSlack)

    
    new_model_lay = rep_weights.solve_optimization_problem(model_lay, cost_expr, gdp_formulation, solver_factory, solver_language)

    model_new_params = rep_weights.set_new_params(new_model_lay)

    
    new_model = rep_weights.return_repaired_model(model_new_params, model_output_type)

    weights = [rep_weights.model_orig_params[iterate] for iterate in range(0,2*(len(architecture)-1),2)]
    bias = [rep_weights.model_orig_params[iterate] for iterate in range(1,2*(len(architecture)-1),2)]

    new_weight = [model_new_params[iterate] for iterate in range(0,2*(len(architecture)-1),2)]
    new_bias = [model_new_params[iterate] for iterate in range(1,2*(len(architecture)-1),2)]

    y_new_train = new_model.predict(x_train)

    y_train_original = model_orig.predict(x_train)
    
    MSE_original_nn_train = cost_function(y_train, y_train_original)/y_train.shape[0]
    MSE_new_nn_train = cost_function(y_train, y_new_train)/y_train.shape[0]

    weight_error = np.max(new_weight[layer_to_repair-1]-weights[layer_to_repair-1])
    bias_error = np.max(new_bias[layer_to_repair-1]-bias[layer_to_repair-1])

    results = Results(weights, bias, new_weight, new_bias, new_model, MSE_original_nn_train, MSE_new_nn_train, weight_error, bias_error)
    return results
