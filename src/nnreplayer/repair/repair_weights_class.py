import numpy as np
from ..utils.utils import mlp_get_weights, mlp_set_weights
from ..form_nn.mlp import MLP
from ..mip.mip_nn_model import MIPNNModel
import pyomo.environ as pyo
from tensorflow import keras


class repair_weights:
    # pylint: disable=invalid-name
    
    """_summary_"""

    def __init__(
        self,
        model_orig,
        layer_to_repair,
        architecture,
        output_constraint_list,
        cost_function_output,
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
        self.layer_to_repair = layer_to_repair
        self.architecture = architecture
        self.cost_function_output = cost_function_output
        self.model_orig_params = mlp_get_weights(self.model_orig)
        self.output_constraint_list = output_constraint_list

    def extract_network(self, x_dataset):
        """_summary_

        Args:
            x_dataset (_type_): _description_

        Returns:
            _type_: _description_
        """

        print("************************************************")
        mlp_orig = MLP(
            self.architecture[0], self.architecture[-1], self.architecture[1:-1]
        )
        mlp_orig = mlp_set_weights(mlp_orig, self.model_orig_params)
        layer_values_train = mlp_orig(x_dataset, relu=False)

        return layer_values_train

    def set_up_optimizer(self, y_train, layer_values_train, weightSlack):
        """_summary_

        Args:
            y_train (_type_): _description_
            layer_values_train (_type_): _description_
            weightSlack (_type_): _description_

        Returns:
            _type_: _description_
        """
        weights = [
            self.model_orig_params[iterate]
            for iterate in range(0, 2 * (len(self.architecture) - 1), 2)
        ]
        bias = [
            self.model_orig_params[iterate]
            for iterate in range(1, 2 * (len(self.architecture) - 1), 2)
        ]

        num_samples = layer_values_train[self.layer_to_repair - 2].shape[0]
        mip_model_layer = MIPNNModel(
            self.layer_to_repair, self.architecture, weights, bias
        )
        y_ = mip_model_layer(
            layer_values_train[self.layer_to_repair - 2],
            (num_samples, self.architecture[self.layer_to_repair - 1]),
            self.output_constraint_list,
            weightSlack=weightSlack,
        )
        model_lay = mip_model_layer.model

        cost_expr = self.cost_function_output(y_, y_train)

        # minimize error bound
        dw_l = "dw"
        cost_expr += getattr(model_lay, dw_l)
        return cost_expr, model_lay

    def solve_optimization_problem(
        self,
        model_lay,
        cost_expr,
        gdp_formulation,
        solver_factory,
        solver_language,
        optimizer_time_limit,
        optimizer_mip_gap,
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
        # gdp_formulation =  'gdp.bigm'
        # solver_factory = 'gurobi'
        # solver_language = "python"
        model_lay.obj = pyo.Objective(expr=cost_expr)
        pyo.TransformationFactory(gdp_formulation).apply_to(model_lay)
        opt = pyo.SolverFactory(solver_factory, solver_io=solver_language)
        opt.options["timelimit"] = optimizer_time_limit
        opt.options["mipgap"] = optimizer_mip_gap
        opt.solve(model_lay, tee=True)
        print(model_lay.dw.display())
        return model_lay

    def set_new_params(self, model_lay):
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
        new_bias = np.zeros((1, self.architecture[self.layer_to_repair]))

        for j in range(self.architecture[self.layer_to_repair]):
            new_bias[0, j] = eval("model_lay.b{}[j].value".format(self.layer_to_repair))
            for i in range(self.architecture[self.layer_to_repair - 1]):

                new_weight[i, j] = eval(
                    "model_lay.w{}[i,j].value".format(self.layer_to_repair)
                )

        # Set new weights and bias
        model_new_params = []
        iterate = 0
        for j in range(len(self.architecture) - 1):
            if j + 1 != self.layer_to_repair:
                model_new_params.append(self.model_orig_params[iterate])
                # print(model_orig_params[iterate].shape)
                iterate = iterate + 1
                model_new_params.append(self.model_orig_params[iterate])
                # print(model_orig_params[iterate].shape)
                iterate = iterate + 1
            else:
                # print(iterate)
                model_new_params.append(new_weight)
                # print(new_weight.shape)

                model_new_params.append(np.squeeze(new_bias))
                iterate = iterate + 2

        return model_new_params

    def return_repaired_model(self, model_new_params, model_output_type):
        """_summary_

        Args:
            model_new_params (_type_): _description_
            model_output_type (_type_): _description_

        Returns:
            _type_: _description_
        """

        if model_output_type == "keras":
            new_model = keras.models.clone_model(self.model_orig)
            weights_bias_iterate = 0
            for iterate in range(len(self.architecture) - 1):
                new_model.layers[iterate].set_weights(
                    model_new_params[weights_bias_iterate : weights_bias_iterate + 2]
                )
                weights_bias_iterate = weights_bias_iterate + 2

        return new_model
