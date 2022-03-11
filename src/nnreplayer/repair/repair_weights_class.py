"""builds a repair model 

Raises:
    ValueError: _description_
    ImportError: _description_
    ValueError: _description_

Returns:
    _type_: _description_
"""
import os
import numpy as np
import pyomo.environ as pyo
from tensorflow import keras
from ..utils.utils import tf2_get_weights, tf2_get_architecture
from ..form_nn.mlp import MLP
from ..mip.mip_nn_model import MIPNNModel


def give_mse_error(data1, data2):
    """return the mean square error of data1-data2 samples

    Args:
        data1 (ndarray): predicted targets
        data2 (ndarray): original targets

    Returns:
        float: mse error
    """
    row, col = np.array(data1).shape
    _squared_sum = 0
    for i in range(row):
        for j in range(col):
            _squared_sum += (data1[i, j] - data2[i, j]) ** 2

    return _squared_sum / row


class NNRepair:
    # pylint: disable=invalid-name

    """Neural Network repair class"""

    def __init__(
        self,
        model_orig,
        # layer_to_repair,
        # architecture,
        # output_constraint_list,
        # cost_function_output,
    ):
        """Creates a 'NNRepair' model instance.

        Args:
            model_orig (keras.engine.sequential.Sequential): input is the keras tf model
        """
        self.model_orig = model_orig
        self.architecture = tf2_get_architecture(self.model_orig)
        self.model_mlp = MLP(
            self.architecture[0], self.architecture[-1], self.architecture[1:-1]
        )
        self.model_mlp.set_mlp_params(tf2_get_weights(self.model_orig))
        self.cost_function_output = give_mse_error
        # self.__compile_flag = False
        self.opt_model = None
        self.layer_to_repair = None
        self.output_constraint_list = []

    def compile(
        self,
        x_repair,
        y_repair,
        layer_2_repair,
        output_constraint_list=None,
        cost=give_mse_error,
        cost_weights=np.array([1.0, 1.0]),
        max_weight_bound=1.0,
    ):
        """_summary_

        Args:
            x_repair (ndarray): input repair samples
            y_repair (ndarray): output repair samples
            layer_2_repair (int): target repair layer
            output_constraint_list (list[nnreplayer.utils.utils.constraints_class], optional): list of output constraints . Defaults to None.
            cost (function, optional): minimization loss function. Defaults to give_mse_error.
            cost_weights (list[ndarray], optional): cost_weights[0]: weight of  min loss, cost_weights[1]: weight of weight bounding slack variable. Defaults to np.array([1.0, 1.0]).
            max_weight_bound (float, optional): upper bound of weights error. Defaults to 1.0.
        """
        # set repair parameters:
        self.layer_to_repair = layer_2_repair
        self.cost_function_output = cost
        self.output_constraint_list = output_constraint_list

        self.__set_up_optimizer(
            y_repair,
            self.extract_network_layers_values(x_repair),
            max_weight_bound,
            cost_weights,
        )

    def repair(self, options):
        """perform the layer-wise repair and updates the weights of model_mlp

        Args:
            options (nnreplayer.utils.options.Options): optimization options
        """
        self.__solve_optimization_problem(
            options.gdp_formulation,
            options.solver_factory,
            options.optimizer_options,
        )
        self.set_new_params()
        return self.return_repaired_model(options.model_output_type)

    def reset(self):
        """reset the model_mlp model to the original model"""
        self.architecture = tf2_get_architecture(self.model_orig)
        self.model_mlp = MLP(
            self.architecture[0], self.architecture[-1], self.architecture[1:-1]
        )
        self.model_mlp.set_mlp_params(tf2_get_weights(self.model_orig))
        self.cost_function_output = give_mse_error
        # self.__compile_flag = False
        self.opt_model = None
        self.layer_to_repair = None
        self.output_constraint_list = []

    def print_opt_model(self, direc=None):
        """print or store the pyomo optimization model

        Args:
            direc (str, optional): directory to print (stdout) the modelled opt. Defaults to None.

        Raises:
            ValueError: returns an error if opt is not complied
        """
        if self.opt_model is None:
            raise ValueError(
                "Optimization model does not exist. First compile the model!"
            )
        if direc is not None:
            if not os.path.exists(direc):
                raise ImportError(f"path {direc} does not exist!")
            with open(
                direc + f"/opt_model_print_lay{self.layer_to_repair}.txt",
                "w",
                encoding="utf8",
            ) as file:
                self.opt_model.pprint(ostream=file)
        else:
            self.opt_model.pprint()

    def extract_network_layers_values(self, x_dataset):
        """extract the values of each layer for all input samples

        Args:
            x_dataset (ndarray): network input samples

        Returns:
            list[ndarray]: values of layers give input samples
        """

        layer_values = self.model_mlp(x_dataset)

        return layer_values

    def set_new_params(self):
        """_summary_

        Args:
            model_lay (_type_): _description_
        """
        new_weight = np.zeros(
            (
                self.architecture[self.layer_to_repair - 1],
                self.architecture[self.layer_to_repair],
            )
        )
        new_bias = np.zeros(self.architecture[self.layer_to_repair])
        for keys, items in (
            getattr(self.opt_model, f"w{self.layer_to_repair}").get_values().items()
        ):
            new_weight[keys[0], keys[1]] = items
        for keys, items in (
            getattr(self.opt_model, f"b{self.layer_to_repair}").get_values().items()
        ):
            new_bias[keys] = items

        self.model_mlp.set_mlp_params_layer(
            [new_weight, new_bias], self.layer_to_repair
        )

    def return_repaired_model(self, model_output_type):
        """_summary_

        Args:
            model_new_params (_type_): _description_
            model_output_type (_type_): _description_

        Returns:
            _type_: _description_
        """
        model_new_params = self.model_mlp.get_mlp_params()
        if model_output_type == "keras":
            new_model = keras.models.clone_model(self.model_orig)
            weights_bias_iterate = 0
            for iterate in range(len(self.architecture) - 1):
                new_model.layers[iterate].set_weights(
                    model_new_params[weights_bias_iterate : weights_bias_iterate + 2]
                )
                weights_bias_iterate = weights_bias_iterate + 2

        return new_model

    def __set_up_optimizer(
        self,
        y_repair,
        layer_values,
        max_weight_bound,
        cost_weights=np.array([1.0, 1.0]),
    ):
        """_summary_

        Args:
            y_train (_type_): _description_
            layer_values_train (_type_): _description_
            max_weight_bound (_type_): _description_

        Returns:
            _type_: _description_
        """
        weights = self.model_mlp.get_mlp_weights()
        bias = self.model_mlp.get_mlp_biases()
        # num_samples = layer_values[self.layer_to_repair - 2].shape[0]
        num_samples = layer_values[self.layer_to_repair - 1].shape[0]
        mip_model_layer = MIPNNModel(
            self.layer_to_repair,
            self.architecture,
            weights,
            bias,
        )
        y_ = mip_model_layer(
            # layer_values[self.layer_to_repair - 2],
            layer_values[self.layer_to_repair - 1],
            (num_samples, self.architecture[self.layer_to_repair - 1]),
            self.output_constraint_list,
            max_weight_bound=max_weight_bound,
        )
        self.opt_model = mip_model_layer.model

        cost_expr = cost_weights[0] * self.cost_function_output(y_, y_repair)
        # minimize error bound
        dw_l = "dw"
        cost_expr += cost_weights[1] * getattr(self.opt_model, dw_l) ** 2
        self.opt_model.obj = pyo.Objective(expr=cost_expr)

    def __solve_optimization_problem(
        self,
        # model_lay,
        # cost_expr,
        gdp_formulation,
        solver_factory,
        optimizer_options,
    ):
        """_summary_

        Args:
            model_lay (_type_): _description_
            cost_expr (_type_): _description_
            gdp_formulation (_type_): _description_
            solver_factory (_type_): _description_
            solver_language (_type_): _description_
            optimizer_time_limit (_type_): _description_
            optimizer_mip_gap (_type_): _description_

        Returns:
            _type_: _description_
        """
        # model_lay.obj = pyo.Objective(expr=cost_expr)
        if self.opt_model is None:
            raise ValueError(
                "Optimization model does not exist. First compile the model!"
            )
        pyo.TransformationFactory(gdp_formulation).apply_to(self.opt_model)
        opt = pyo.SolverFactory(solver_factory, solver_io="python")
        for key in optimizer_options:
            opt.options[key] = optimizer_options[key]
        opt.solve(self.opt_model, tee=True)
        print(self.opt_model.dw.display())
