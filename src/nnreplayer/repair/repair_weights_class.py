import numpy as np
import pyomo.environ as pyo
from tensorflow import keras
from ..utils.utils import tf2_get_weights, tf2_get_architecture
from ..form_nn.mlp import MLP
from ..mip.mip_nn_model import MIPNNModel


def give_mse_error(data1, data2):
    """_summary_

    Args:
        data1 (_type_): _description_
        data2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    row, col = np.array(data1).shape
    _squared_sum = 0
    for i in range(row):
        for j in range(col):
            _squared_sum += (data1[i, j] - data2[i, j]) ** 2

    return _squared_sum / row


class repair_weights:
    # pylint: disable=invalid-name

    """_summary_"""

    def __init__(
        self,
        model_orig,
        # layer_to_repair,
        # architecture,
        # output_constraint_list,
        # cost_function_output,
    ):
        """_summary_

        Args:
            model_orig (_type_): _description_
            layer_to_repair (_type_): _description_
            architecture (_type_): _description_
            output_constraint_list (_type_): _description_
            cost_function_output (_type_): _description_
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
            x_dataset (_type_): _description_
            layer_2_repair (_type_): _description_
            output_constraint_list (_type_): _description_
            cost (_type_, optional): _description_. Defaults to give_mse_error.
            cost_weights (list, optional): _description_. Defaults to [1.0, 1.0].
            max_weight_bound (float, optional): _description_. Defaults to 1.0.
        """
        # set repair parameters:
        self.layer_to_repair = layer_2_repair
        self.cost_function_output = cost
        self.output_constraint_list = output_constraint_list

        self.__set_up_optimizer(
            y_repair,
            self.extract_network_values(x_repair),
            max_weight_bound,
            cost_weights,
        )

    def repair(self, options):
        """_summary_

        Args:
            options (_type_): _description_
        """
        self.__solve_optimization_problem(
            options.gdp_formulation,
            options.solver_factory,
            options.optimizer_options,
        )
        self.set_new_params()
        return self.return_repaired_model(options.model_output_type)

    def reset(self):
        """reset the mlp model to the original model"""
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

    # def update_mlp_model(self):

    def extract_network_values(self, x_dataset):
        """_summary_

        Args:
            x_dataset (_type_): _description_

        Returns:
            _type_: _description_
        """

        print("************************************************")
        # mlp_orig = MLP(
        #     self.architecture[0], self.architecture[-1], self.architecture[1:-1]
        # )
        # mlp_orig.set_mlp_params(self.model_orig_params)
        layer_values = self.model_mlp(x_dataset, relu=False)

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
        # Set new weights and bias
        # model_new_params = []
        # iterate = 0
        # for j in range(len(self.architecture) - 1):
        #     if j + 1 != self.layer_to_repair:
        #         model_new_params.append(self.model_orig_params[iterate])
        #         # print(model_orig_params[iterate].shape)
        #         iterate = iterate + 1
        #         model_new_params.append(self.model_orig_params[iterate])
        #         # print(model_orig_params[iterate].shape)
        #         iterate = iterate + 1
        #     else:
        #         # print(iterate)
        #         model_new_params.append(new_weight)
        #         # print(new_weight.shape)

        #         model_new_params.append(np.squeeze(new_bias))
        #         iterate = iterate + 2

        # return model_new_params

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
        # weights = [
        #     self.model_orig_params[iterate]
        #     for iterate in range(0, 2 * (len(self.architecture) - 1), 2)
        # ]
        # bias = [
        #     self.model_orig_params[iterate]
        #     for iterate in range(1, 2 * (len(self.architecture) - 1), 2)
        # ]
        weights = self.model_mlp.get_mlp_weights()
        bias = self.model_mlp.get_mlp_biases()
        num_samples = layer_values[self.layer_to_repair - 2].shape[0]
        mip_model_layer = MIPNNModel(
            self.layer_to_repair,
            self.architecture,
            weights,
            bias,
        )
        y_ = mip_model_layer(
            layer_values[self.layer_to_repair - 2],
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
