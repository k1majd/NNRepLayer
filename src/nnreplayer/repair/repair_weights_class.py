from typing import Any, Union, Type, Optional, Type, List, Callable

from ..utils.options import Options
from ..utils.utils import constraints_class
import os
import numpy as np
import pyomo.environ as pyo
from tensorflow import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.utils import tf2_get_weights, tf2_get_architecture
from ..utils.utils import pt_get_weights, pt_get_architecture
from ..form_nn.mlp import MLP
from ..mip.mip_nn_model import MIPNNModel
from ..utils.utils import give_mse_error
import numpy.typing as npt


class NNRepair:
    # pylint: disable=invalid-name
    """NN

    Attributes:
        model_orig
        architecture
        model_mlp
        cost_function_output
        data_precision
        param_precision
        opt_model
        layer_to_repair
        output_name
        num_samples
        output_constraint_list

    Methods:


    """

    def __init__(self, model_orig, model_type="tensorflow") -> None:
        """Initializes the NNRepair Class

        Args:
            model_orig (_type_): Original Model intended to be repaired.
        """
        assert (
            model_orig is not None
        ), f"Expected Model. Received {type(model_orig)} instead."
        self.model_orig = model_orig
        self.model_type = model_type
        if model_type == "tensorflow":
            self.architecture = tf2_get_architecture(self.model_orig)
            self.model_mlp = MLP(
                self.architecture[0],
                self.architecture[-1],
                self.architecture[1:-1],
            )
            self.model_mlp.set_mlp_params(tf2_get_weights(self.model_orig))
        elif model_type == "pytorch":

            self.architecture = pt_get_architecture(self.model_orig)
            self.model_mlp = MLP(
                self.architecture[0],
                self.architecture[-1],
                self.architecture[1:-1],
            )
            self.model_mlp.set_mlp_params(pt_get_weights(self.model_orig))
        else:
            raise TypeError(
                f"Expected tensorflow or pytorch. Received {model_type} instead."
            )
        self.cost_function_output = give_mse_error
        self.data_precision = None
        self.param_precision = None
        self.opt_model = None
        self.layer_to_repair = None
        self.output_name = None
        self.num_samples = None
        self.output_constraint_list = []

    def compile(
        self,
        x_repair: npt.NDArray,
        y_repair: npt.NDArray,
        layer_2_repair: int,
        output_constraint_list: Optional[List[Type[constraints_class]]] = None,
        cost: Callable = give_mse_error,
        cost_weights: npt.NDArray = np.array([1.0, 1.0]),
        max_weight_bound: Union[int, float] = 1.0,
        data_precision: int = 4,
        param_precision: int = 4,
    ) -> None:
        """Compile the optimization model and setup the repair optimizer

        Args:
            x_repair (npt.NDArray): Input repair samples
            y_repair (npt.NDArray): Output repair samples
            layer_2_repair (int): Target repair layer
            output_constraint_list (Optional[list[Type[constraints_class]]], optional): List of output constraints Defaults to None.
            cost (function, optional): Minimization loss function. Defaults to give_mse_error.

            cost_weights (npt.NDArray, optional): cost_weights[0]: weight of  min loss,
                                                                        cost_weights[1]: weight of weight bounding slack variable.
                                                                                Defaults to np.array([1.0, 1.0]).
            max_weight_bound (Union, optional): Upper bound of weights errorDefaults to 1.0.
            data_precision (int, optional): precision of rounding to decimal place for dataDefaults to 4.
            param_precision (int, optional): precision of rounding to decimal place for parameters. Defaults to 4.
        """

        # set repair parameters:
        if x_repair.shape[-1] != self.architecture[0]:
            raise TypeError(
                f"Input Set Mismatch. Expected (X, {self.architecture[0]}). Received (X, {x_repair.shape[-1]} instead."
            )
        if y_repair.shape[-1] != self.architecture[-1]:
            raise TypeError(
                f"Input Set Mismatch. Expected (X, {self.architecture[-1]}). Received (X, {y_repair.shape[-1]} instead."
            )
        if not (
            layer_2_repair <= len(self.architecture) - 1
            and layer_2_repair >= 1
        ):
            raise TypeError(
                f"Layer to repair out of bounds. Expected [{1}, {len(self.architecture)-1}]. Received {layer_2_repair} instead."
            )

        self.layer_to_repair = layer_2_repair
        self.cost_function_output = cost
        self.output_constraint_list = output_constraint_list
        self.data_precision = data_precision
        self.param_precision = param_precision

        self.__set_up_optimizer(
            np.round(y_repair, data_precision),
            self.extract_network_layers_values(
                np.round(x_repair, data_precision)
            ),
            max_weight_bound,
            cost_weights,
        )

    def repair(self, options: Type[Options]) -> Any:
        """Perform the layer-wise repair and update the weights of model_mlp

        Args:
            options (Type[Options]): optimization options

        Returns:
            Any: _description_
        """

        self.__solve_optimization_problem(
            options.gdp_formulation,
            options.solver_factory,
            options.optimizer_options,
        )
        self.__set_new_params()
        repaired_model = self.__return_repaired_model()
        return repaired_model

    def reset(self):
        """Reset the model_mlp model to the original model"""

        self.architecture = tf2_get_architecture(self.model_orig)
        self.model_mlp = MLP(
            self.architecture[0],
            self.architecture[-1],
            self.architecture[1:-1],
        )
        self.model_mlp.set_mlp_params(tf2_get_weights(self.model_orig))
        self.cost_function_output = give_mse_error
        self.opt_model = None
        self.layer_to_repair = None
        self.output_name = None
        self.num_samples = None
        self.output_constraint_list = []

    def summary(self, direc: Optional[str] = None):
        """Print and/or store the pyomo optimization model

        Args:
            direc (str, optional): directory to print (stdout) the modelled opt. Defaults to None.

        Raises:
            ValueError: Raises an error if opt is not complied
        """
        if self.opt_model is None:
            raise ValueError(
                "Optimization model does not exist. First compile the model!"
            )
        if direc is not None:
            if not os.path.exists(direc):
                raise ImportError(f"path {direc} does not exist!")
            with open(
                direc + f"/opt_model_summery_layer{self.layer_to_repair}.txt",
                "w",
                encoding="utf8",
            ) as file:
                self.opt_model.pprint(ostream=file)
        else:
            self.opt_model.pprint()

    def extract_network_layers_values(
        self, x_dataset: npt.NDArray
    ) -> List[npt.NDArray]:
        """Extract the values of each layer for all input samples

        Args:
            x_dataset (NDArray): network input samples

        Returns:
            list[NDArray]: values of layers give input samples
        """

        assert (
            x_dataset.shape[-1] == self.architecture[0]
        ), f"Input Set Mismatch. Expected (X, {self.architecture[0]}). Received (X, {x_dataset.shape[-1]} instead."

        layer_values = self.model_mlp(x_dataset)

        return layer_values

    def __return_repaired_model(self):
        # pylint: disable=pointless-string-statement
        """Returns the repaired model in the given format"""

        """
        TODO
        add support for the other types of models: hdf5, ...
        """
        model_new_params = self.model_mlp.get_mlp_params()
        print("Hello")
        if self.model_type == "tensorflow":
            new_model = keras.models.clone_model(self.model_orig)
            weights_bias_iterate = 0
            for iterate in range(len(self.architecture) - 1):
                new_model.layers[iterate].set_weights(
                    model_new_params[
                        weights_bias_iterate : weights_bias_iterate + 2
                    ]
                )
                weights_bias_iterate = weights_bias_iterate + 2

        elif self.model_type == "pytorch":
            import copy

            new_model = copy.deepcopy(self.model_orig)
            weights_bias_iterate = 0
            for name, param in new_model.named_parameters():

                old_param = param.data.numpy()

                new_param = model_new_params[weights_bias_iterate]
                if "weight" in set(name.split(".")):
                    new_param = new_param.T
                if old_param.shape != new_param.shape:
                    raise ValueError("Check model params.")
                print("*****************************")
                print(name)
                print(new_param)
                # new_model.state_dict()[lay] = torch.Tensor(new_param)
                param.data = torch.Tensor(new_param)
                print(new_model.state_dict()[name])
                weights_bias_iterate += 1
            # for x in new_model.state_dict():
            #     print(new_model.state_dict()[x])
            new_model.eval()
        else:
            raise TypeError(
                f"Expected tensorflow or pytorch. Received {self.model_type} instead."
            )
        return new_model

    def __set_new_params(self):
        """Update the weight and bias terms of model_mlp for the target layer"""

        new_weight = np.zeros(
            (
                self.architecture[self.layer_to_repair - 1],
                self.architecture[self.layer_to_repair],
            )
        )
        new_bias = np.zeros(self.architecture[self.layer_to_repair])
        for keys, items in (
            getattr(self.opt_model, f"w{self.layer_to_repair}")
            .get_values()
            .items()
        ):
            new_weight[keys[0], keys[1]] = items
        for keys, items in (
            getattr(self.opt_model, f"b{self.layer_to_repair}")
            .get_values()
            .items()
        ):
            new_bias[keys] = items

        self.model_mlp.set_mlp_params_layer(
            [new_weight, new_bias], self.layer_to_repair
        )

    def __set_up_optimizer(
        self,
        y_repair: npt.NDArray,
        layer_values: List[npt.NDArray],
        max_weight_bound: Union[int, float],
        cost_weights: npt.NDArray = np.array([1.0, 1.0]),
    ) -> None:
        """_summary_

        Args:
            y_repair (npt.NDArray): _description_
            layer_values (list[npt.NDArray]): _description_
            max_weight_bound (Union[int, float]): _description_
            cost_weights (npt.NDArray, optional): _description_. Defaults to np.array([1.0, 1.0]).
        """

        if y_repair.shape[-1] != self.architecture[-1]:
            raise TypeError(
                f"Input Set Mismatch. Expected (X, {self.architecture[-1]}). Received (X, {y_repair.shape[-1]} instead."
            )
        weights = self.model_mlp.get_mlp_weights()
        bias = self.model_mlp.get_mlp_biases()

        # specify the precision of weights, bias, and layer values
        for l, w in enumerate(weights):
            weights[l] = np.round(w, self.param_precision)
        for l, b in enumerate(bias):
            bias[l] = np.round(b, self.param_precision)
        for l, value in enumerate(layer_values):
            layer_values[l] = np.round(value, self.data_precision)

        self.num_samples = layer_values[self.layer_to_repair - 1].shape[0]
        mip_model_layer = MIPNNModel(
            self.layer_to_repair,
            self.architecture,
            weights,
            bias,
            param_bounds=(
                np.round(
                    np.min(
                        [
                            np.min(bias[self.layer_to_repair - 1]),
                            np.min(weights[self.layer_to_repair - 1]),
                        ]
                    )
                    - max_weight_bound
                    - 0.01,
                    self.param_precision,
                ),
                np.round(
                    np.max(
                        [
                            np.max(bias[self.layer_to_repair - 1]),
                            np.max(weights[self.layer_to_repair - 1]),
                        ]
                    )
                    + max_weight_bound
                    + 0.01,
                    self.data_precision,
                ),
            ),
        )
        y_ = mip_model_layer(
            # layer_values[self.layer_to_repair - 2],
            layer_values[self.layer_to_repair - 1],
            (self.num_samples, self.architecture[self.layer_to_repair - 1]),
            self.output_constraint_list,
            max_weight_bound=max_weight_bound,
            # output_bounds=(np.min(y_repair), np.max(y_repair)),
        )
        self.output_name = y_.name
        self.opt_model = mip_model_layer.model

        cost_expr = cost_weights[0] * self.cost_function_output(y_, y_repair)
        # minimize error bound
        dw_l = "dw"
        cost_expr += cost_weights[1] * getattr(self.opt_model, dw_l) ** 2
        self.opt_model.obj = pyo.Objective(expr=cost_expr)

    def __solve_optimization_problem(
        self,
        gdp_formulation: str,
        solver_factory: str,
        optimizer_options: dict,
    ) -> None:
        """_summary_

        Args:
            gdp_formulation (str): _description_
            solver_factory (str): _description_
            optimizer_options (dict): _description_

        Raises:
            ValueError: _description_
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
        print("----------------------")
        print(self.opt_model.dw.display())
