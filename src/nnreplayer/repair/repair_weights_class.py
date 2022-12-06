from typing import Any, Union, Optional, Type, List, Callable
import os
import copy
import torch
import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo
from tensorflow import keras
from nnreplayer.utils import tf2_get_weights, tf2_get_architecture
from nnreplayer.utils import pt_get_weights, pt_get_architecture
from nnreplayer.form_nn import MLP
from nnreplayer.mip import MIPNNModel
from nnreplayer.mip import MIPNNModelExtend
from nnreplayer.lp_bound import LPNNModel
from nnreplayer.utils import give_mse_error
from nnreplayer.utils import Options
from nnreplayer.utils import ConstraintsClass, BoundStatTracker


class NNRepair:
    """NN Repair class helps in initializing, compiling ang running the repair process."""

    def __init__(self, model_orig, model_type="tensorflow") -> None:
        """Initialize the repair process.

        Args:
            model_orig: Original Model intended for repair.
            model_type: Type of IInput Model (Pytorch or Tensorflow). Defaults to "tensorflow".

        Raises:
            TypeError: _description_
        """

        assert (
            model_orig is not None
        ), f"Expected Model. Received {type(model_orig)} instead."

        self.model_orig = model_orig
        self.model_type = model_type
        self.model_mlp, self.architecture = self.__give_mlp_model(
            model_orig, model_type
        )
        self.cost_function_output = give_mse_error
        self.data_precision = None
        self.param_precision = None
        self.opt_model = None
        self.layer_to_repair = None
        self.output_name = None
        self.output_variable = None
        self.num_samples = None
        self.output_constraint_list = []
        self.repair_node_list = []
        self.__target_original_weights = []
        self.__target_original_bias = []
        self.extended_model_mlp_blocks = []
        self.ub_last_layer = None
        self.lb_last_layer = None
        self.output_state_name = None
        self.output_state_variable = None

    def compile(
        self,
        x_repair: npt.NDArray,
        # y_repair: npt.NDArray,
        layer_2_repair: int,
        output_constraint_list: Optional[List[Type[ConstraintsClass]]] = None,
        cost: Callable = give_mse_error,
        # cost_weights: npt.NDArray = np.array([1.0, 1.0]),
        max_weight_bound: Union[int, float] = 1.0,
        data_precision: int = 4,
        param_precision: int = 4,
        repair_node_list: List[int] = None,
        bound_tightening_method: str = "ia",
        w_error_norm: int = 0,
        param_bounds: tuple = None,
        output_bounds: tuple = None,
    ) -> None:

        """Compile the optimization model and setup the repair optimizer

        Args:
            x_repair : Input repair samples
            y_repair : Output repair samples
            layer_2_repair : Target repair layer
            output_constraint_list: List of output constraints Defaults to None.
            cost: Minimization loss function. Defaults to give_mse_error.
            max_weight_bound: Upper bound of weights errorDefaults to 1.0.
            data_precision: precision of rounding to decimal place for dataDefaults to 4.
            param_precision: precision of rounding to decimal place for parameters. Defaults to 4.
            repair_node_list: List of indices of target repair nodes. Defaults to None.
            bound_tightening_method: Method to use for bound tightening. Defaults to "ia", options are "ia" or "lp".
            w_error_norm (int, optional): weight error norm type 0 = L-inf, 1 = L-1. Defaults to 0.
            param_bounds: bounds of parameters. Defaults to None.
            output_bounds: bounds of output. Defaults to None.
        """

        # set repair parameters:
        if x_repair.shape[-1] != self.architecture[0]:
            raise TypeError(
                f"Input Set Mismatch. Expected (X, {self.architecture[0]}). Received (X, {x_repair.shape[-1]} instead."
            )
        # if y_repair.shape[-1] != self.architecture[-1]:
        #     raise TypeError(
        #         f"Input Set Mismatch. Expected (X, {self.architecture[-1]}). Received (X, {y_repair.shape[-1]} instead."
        #     )
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
        if (
            repair_node_list is None
            or len(repair_node_list) == 0
            or self.layer_to_repair == len(self.architecture) - 1
        ):
            repair_node_list = list(range(self.architecture[layer_2_repair]))
        self.repair_node_list = repair_node_list
        self.__set_up_optimizer(
            # np.round(y_repair, data_precision),
            x_repair,
            max_weight_bound,
            # cost_weights,
            bound_tightening_method,
            w_error_norm,
            param_bounds,
            output_bounds,
        )

    def repair(
        self,
        options: Type[Options],
        y_repair: npt.NDArray,
        cost_weights: npt.NDArray = np.array([1.0, 1.0]),
    ) -> Any:

        """Perform the layer-wise repair and update the weights of model_mlp

        Args:
            options: optimization options

        Returns:
            Repaired Model.
        """
        self.__specify_cost(y_repair, cost_weights)
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
        self.output_variable = None
        self.num_samples = None
        self.output_constraint_list = []
        ######################################
        # TODO: add this part
        self.repair_node_list = []
        self.__target_original_weights = []
        self.__target_original_bias = []
        self.extended_model_mlp_blocks = []
        self.ub_last_layer = None
        self.lb_last_layer = None
        self.output_state_name = None
        self.output_state_variable = Node

    def extend(
        self,
        new_model_block,
        input_order: List[str],
        x_repair: npt.NDArray,
        output_constraint_list: Optional[List[Type[ConstraintsClass]]] = None,
    ):
        """extend the opt model"""
        self.extended_model_mlp_blocks.append(
            self.__give_mlp_model(new_model_block, self.model_type)
        )

        weights = self.extended_model_mlp_blocks[-1][0].get_mlp_weights()
        bias = self.extended_model_mlp_blocks[-1][0].get_mlp_biases()
        for i, w in enumerate(weights):
            weights[i] = np.round(w, self.param_precision)
        for i, b in enumerate(bias):
            bias[i] = np.round(b, self.param_precision)

        ub_mat, lb_mat = self.__tight_bounds_ia_extend(
            weights, bias, x_repair, input_order
        )
        mip_model_layer = MIPNNModelExtend(
            self.extended_model_mlp_blocks[-1][1],
            weights,
            bias,
            self.opt_model,
            last_ctrl_layer=len(self.architecture),
            layer_num=len(self.architecture) + 1,
            param_precision=self.param_precision,
        )
        y_ = mip_model_layer(
            x_repair,
            (self.num_samples, x_repair.shape[1] + self.architecture[-1]),
            output_constraint_list,
            input_order,
            self.output_name,
            nodes_upper=ub_mat,
            nodes_lower=lb_mat,
        )
        self.output_state_name = y_.name
        self.output_state_variable = y_
        self.opt_model = mip_model_layer.model

    def summary(self, direc: Optional[str] = None):

        """Print and/or store the pyomo optimization model

        Args:
            direc: directory to print (stdout) the modelled opt. Defaults to None.

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
            x_dataset: network input samples

        Returns:
            values of layers give input samples
        """

        assert (
            x_dataset.shape[-1] == self.architecture[0]
        ), f"Input Set Mismatch. Expected (X, {self.architecture[0]}). Received (X, {x_dataset.shape[-1]} instead."

        layer_values = self.model_mlp(x_dataset)

        return layer_values

    def __return_repaired_model(self):
        # pylint: disable=pointless-string-statement
        """Returns the repaired model in the given format"""

        model_new_params = self.model_mlp.get_mlp_params()
        # print("Hello")
        if self.model_type == "tensorflow":
            new_model = keras.models.clone_model(self.model_orig)
            weights_bias_iterate = 0
            
            if len(new_model.layers[0].get_weights()) == 0:
                iterate = 1
            else:
                iterate = 0

            for _ in range(len(self.architecture) - 1):
                new_model.layers[iterate].set_weights(
                    model_new_params[
                        weights_bias_iterate : weights_bias_iterate + 2
                    ]
                )
                weights_bias_iterate = weights_bias_iterate + 2
                iterate += 1

        elif self.model_type == "pytorch":

            new_model = copy.deepcopy(self.model_orig)
            weights_bias_iterate = 0
            for name, param in new_model.named_parameters():

                old_param = param.data.numpy()

                new_param = model_new_params[weights_bias_iterate]
                if "weight" in set(name.split(".")):
                    new_param = new_param.T
                if old_param.shape != new_param.shape:
                    raise ValueError("Check model params.")
                # print("*****************************")
                # print(name)
                # print(new_param)
                # new_model.state_dict()[lay] = torch.Tensor(new_param)
                param.data = torch.Tensor(new_param)
                # print(new_model.state_dict()[name])
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
        # extract repaired weight and bias terms

        for c in range(self.architecture[self.layer_to_repair]):
            if c in self.repair_node_list:
                new_bias[c] = getattr(
                    self.opt_model, f"b{self.layer_to_repair}"
                ).get_values()[c]
                for r in range(self.architecture[self.layer_to_repair - 1]):
                    new_weight[r, c] = getattr(
                        self.opt_model, f"w{self.layer_to_repair}"
                    ).get_values()[(r, c)]
            else:
                new_bias[c] = self.__target_original_bias[c]
                for r in range(self.architecture[self.layer_to_repair - 1]):
                    new_weight[r, c] = self.__target_original_weights[r, c]
        # for keys, items in (
        #     getattr(self.opt_model, f"w{self.layer_to_repair}")
        #     .get_values()
        #     .items()
        # ):
        #     new_weight[keys[0], keys[1]] = items
        # for keys, items in (
        #     getattr(self.opt_model, f"b{self.layer_to_repair}")
        #     .get_values()
        #     .items()
        # ):
        #     new_bias[keys] = items

        # specify the untouched weight and bias values in the target layer

        self.model_mlp.set_mlp_params_layer(
            [new_weight, new_bias], self.layer_to_repair
        )

    def __set_up_optimizer(
        self,
        # y_repair: npt.NDArray,
        x_repair: npt.NDArray,
        max_weight_bound: Union[int, float],
        # cost_weights: npt.NDArray = np.array([1.0, 1.0]),
        bound_tightening_method: str = "lp",
        w_error_norm: int = 0,
        param_bounds: tuple = None,
        output_bounds: tuple = None,
    ) -> None:
        """Setting Up optimizer

        Args:
            y_repair: Output Data
            layer_values: Values of ReLU layers after forward pass.
            max_weight_bound: Max Weight Bound
            w_error_norm (int, optional): weight error norm type 0 = L-inf, 1 = L-1. Defaults to 0.

        """

        weights = self.model_mlp.get_mlp_weights()
        bias = self.model_mlp.get_mlp_biases()
        layer_values = self.extract_network_layers_values(x_repair)
        for i, w in enumerate(weights):
            weights[i] = np.round(w, self.param_precision)
        for i, b in enumerate(bias):
            bias[i] = np.round(b, self.param_precision)
        for i, value in enumerate(layer_values):
            layer_values[i] = np.round(value, self.data_precision)

        self.num_samples = layer_values[self.layer_to_repair - 1].shape[0]

        ub_mat, lb_mat = self.__get_node_bounds(
            layer_values,
            weights,
            bias,
            max_weight_bound,
            bound_tightening_method,
        )

        # Update the object with the bounds of last layer
        self.ub_last_layer = ub_mat[-1]
        self.lb_last_layer = lb_mat[-1]

        self.__target_original_weights = weights[self.layer_to_repair - 1]
        self.__target_original_bias = bias[self.layer_to_repair - 1]
        mip_model_layer = MIPNNModel(
            self.layer_to_repair,
            self.architecture,
            weights,
            bias,
            self.repair_node_list,
            w_error_norm,
            max_weight_bound,
            self.param_precision,
        )
        y_ = mip_model_layer(
            layer_values[self.layer_to_repair - 1],
            (self.num_samples, self.architecture[self.layer_to_repair - 1]),
            self.output_constraint_list,
            nodes_upper=ub_mat,
            nodes_lower=lb_mat,
            max_weight_bound=max_weight_bound,
        )
        # specify output variable and initialize the opt_model
        self.output_name = y_.name
        self.output_variable = y_
        self.opt_model = mip_model_layer.model

    def __specify_cost(
        self,
        y_repair: npt.NDArray,
        cost_weights: npt.NDArray = np.array([1.0, 1.0]),
    ) -> None:
        """Specify the Cost Function if MIP model

        Args:
            y_repair: Output Data
            cost_weights (optional): Cost Weights. Defaults to np.array([1.0, 1.0]).

        Raises:
            TypeError: Mismatch between Input and Output Set.
        """
        if y_repair.shape[-1] != (
            self.architecture[-1] + self.extended_model_mlp_blocks[-1][1][-1]
        ):
            raise TypeError(
                f"Input Set Mismatch. Expected (X, {self.architecture[-1]}). Received (X, {y_repair.shape[-1]} instead."
            )
        cost_expr = cost_weights[0] * self.cost_function_output(
            self.output_variable,
            np.round(y_repair[:, 0:1], self.data_precision),
        ) + cost_weights[0] * self.cost_function_output(
            self.output_state_variable,
            np.round(y_repair[:, 1:], self.data_precision),
        )
        # cost_expr = cost_weights[0] * self.cost_function_output(
        #     self.output_state_variable,
        #     np.round(y_repair[:, 1:], self.data_precision),
        # )
        # minimize error bound
        dw_l = "dw"
        db_l = "db"
        if len(getattr(self.opt_model, dw_l)) == 1:
            cost_expr += cost_weights[1] * getattr(self.opt_model, dw_l)
        else:
            for item in getattr(self.opt_model, dw_l)._data.items():
                cost_expr += cost_weights[1] * item[1]
            for item in getattr(self.opt_model, db_l)._data.items():
                cost_expr += cost_weights[1] * item[1]

        self.opt_model.obj = pyo.Objective(expr=cost_expr)

    def __get_node_bounds(
        self,
        layer_values,
        weights,
        bias,
        max_weight_bound,
        bound_tightening_method,
    ):
        print(" ")
        print(f"----------------------------------------")
        print(f"Calculating tight bounds over the nodes")
        print(f"----------------------------------------")
        print(" ")
        print("-> IA method")
        print(" ")
        ub_mat, lb_mat = self.__tight_bounds_ia(
            layer_values,
            weights,
            bias,
            max_weight_bound,
            self.repair_node_list,
        )
        # specify the precision of upper and lower bounds
        for l, ub in enumerate(ub_mat):
            ub_mat[l] = np.round(ub, self.data_precision)
        for l, lb in enumerate(lb_mat):
            lb_mat[l] = np.round(lb, self.data_precision)
        if bound_tightening_method == "lp":
            if self.layer_to_repair < len(self.architecture) - 1:
                print(" ")
                print("-> LP method")
                print(" ")
                ub_mat, lb_mat = self.__tight_bounds_lp(
                    layer_values,
                    weights,
                    bias,
                    ub_mat,
                    lb_mat,
                    max_weight_bound,
                )
            print(" ")
        return ub_mat, lb_mat

    def __tight_bounds_ia_extend(  # temporary method
        self,
        weights,
        bias,
        x_state,
        input_order,
    ):
        architecture = self.extended_model_mlp_blocks[-1][1]
        num_samples = x_state.shape[0]
        current_layer = 1
        # specify control and state ranges
        if input_order.index("state") == 0:
            state_range = [0, x_state.shape[1]]
            control_range = [
                x_state.shape[1],
                x_state.shape[1] + self.architecture[-1],
            ]
        else:
            control_range = [0, self.architecture[-1]]
            state_range = [
                self.architecture[-1],
                self.architecture[-1] + x_state.shape[1],
            ]
        bound_stat_tracker = BoundStatTracker(architecture)
        ub_mat = []
        lb_mat = []
        ub = np.zeros((x_state.shape[0], architecture[current_layer]))
        lb = np.zeros((x_state.shape[0], architecture[current_layer]))

        # First layer bounds
        for node_next in range(architecture[1]):
            for s in range(num_samples):
                lb[s, node_next] = bias[current_layer - 1][node_next]
                ub[s, node_next] = bias[current_layer - 1][node_next]
                for node_prev in range(control_range[0], control_range[1]):
                    w_temp = weights[current_layer - 1][node_prev][node_next]
                    lb[s, node_next] += self.lb_last_layer[s][
                        node_prev - control_range[0]
                    ] * max(w_temp, 0) + self.ub_last_layer[s][
                        node_prev - control_range[0]
                    ] * min(
                        w_temp, 0
                    )
                    ub[s, node_next] += self.ub_last_layer[s][
                        node_prev - control_range[0]
                    ] * max(w_temp, 0) + self.lb_last_layer[s][
                        node_prev - control_range[0]
                    ] * min(
                        w_temp, 0
                    )
                for node_prev in range(state_range[0], state_range[1]):
                    w_temp = weights[current_layer - 1][node_prev][node_next]
                    lb[s, node_next] += (
                        w_temp * x_state[s][node_prev - state_range[0]]
                    )
                    ub[s, node_next] += (
                        w_temp * x_state[s][node_prev - state_range[0]]
                    )
                # update stats
                bound_stat_tracker.update_stats(
                    lb[s, node_next],
                    ub[s, node_next],
                    1,
                    node_next,
                )
        # print stats
        bound_stat_tracker.print_stats(current_layer)

        ub_mat.append(ub)
        lb_mat.append(lb)

        # get the intervals for the subsequent layers
        for layer in range(current_layer + 1, len(architecture)):
            ub = np.zeros((num_samples, architecture[layer]))
            lb = np.zeros((num_samples, architecture[layer]))
            for node_next in range(architecture[layer]):
                for s in range(num_samples):
                    lb[s, node_next] = bias[layer - 1][node_next]
                    ub[s, node_next] = bias[layer - 1][node_next]
                    for node_prev in range(architecture[layer - 1]):
                        w_temp = weights[layer - 1][node_prev][node_next]
                        lb[s, node_next] += lb_mat[layer - 2][s][
                            node_prev
                        ] * max(w_temp, 0) + ub_mat[layer - 2][s][
                            node_prev
                        ] * min(
                            w_temp, 0
                        )
                        ub[s, node_next] += ub_mat[layer - 2][s][
                            node_prev
                        ] * max(w_temp, 0) + lb_mat[layer - 2][s][
                            node_prev
                        ] * min(
                            w_temp, 0
                        )
                    # update stats
                    bound_stat_tracker.update_stats(
                        lb[s, node_next],
                        ub[s, node_next],
                        layer,
                        node_next,
                    )
            # print stats
            bound_stat_tracker.print_stats(layer)
            ub_mat.append(ub)
            lb_mat.append(lb)

        return ub_mat, lb_mat

    def __tight_bounds_ia(
        self,
        layer_values: List[npt.NDArray],
        weights: List[npt.NDArray],
        bias: List[npt.NDArray],
        max_weight_bound: Union[int, float],
        repair_node_list: List[int],
    ) -> tuple[List[npt.NDArray], List[npt.NDArray]]:
        """Returns the active status of the nodes from the layer_to_repair to the end."""
        bound_stat_tracker = BoundStatTracker(self.architecture)
        ub_mat = []
        lb_mat = []
        # get the intervals for the layer_to_repair layer outputs
        ub = np.zeros(
            (layer_values[0].shape[0], self.architecture[self.layer_to_repair])
        )
        lb = np.zeros(
            (layer_values[0].shape[0], self.architecture[self.layer_to_repair])
        )
        for node_next in range(self.architecture[self.layer_to_repair]):
            for s in range(layer_values[0].shape[0]):
                if node_next in repair_node_list:
                    lb[s, node_next] = (
                        bias[self.layer_to_repair - 1][node_next]
                        - max_weight_bound
                    )
                    ub[s, node_next] = (
                        bias[self.layer_to_repair - 1][node_next]
                        + max_weight_bound
                    )
                    for node_prev in range(
                        self.architecture[self.layer_to_repair - 1]
                    ):
                        lb[s, node_next] += (
                            weights[self.layer_to_repair - 1][node_prev][
                                node_next
                            ]
                            - max_weight_bound
                        ) * max(
                            0.0,
                            layer_values[self.layer_to_repair - 1][s][
                                node_prev
                            ],
                        ) + (
                            weights[self.layer_to_repair - 1][node_prev][
                                node_next
                            ]
                            + max_weight_bound
                        ) * min(
                            0.0,
                            layer_values[self.layer_to_repair - 1][s][
                                node_prev
                            ],
                        )

                        ub[s, node_next] += (
                            weights[self.layer_to_repair - 1][node_prev][
                                node_next
                            ]
                            + max_weight_bound
                        ) * max(
                            0.0,
                            layer_values[self.layer_to_repair - 1][s][
                                node_prev
                            ],
                        ) + (
                            weights[self.layer_to_repair - 1][node_prev][
                                node_next
                            ]
                            - max_weight_bound
                        ) * min(
                            0.0,
                            layer_values[self.layer_to_repair - 1][s][
                                node_prev
                            ],
                        )
                    # update stats
                    bound_stat_tracker.update_stats(
                        lb[s, node_next],
                        ub[s, node_next],
                        self.layer_to_repair,
                        node_next,
                    )
                else:
                    lb[s, node_next] = layer_values[self.layer_to_repair][s][
                        node_next
                    ]
                    ub[s, node_next] = layer_values[self.layer_to_repair][s][
                        node_next
                    ]
        # print stats
        bound_stat_tracker.print_stats(self.layer_to_repair)

        ub_mat.append(ub)
        lb_mat.append(lb)

        # get the intervals for the subsequent layers
        for layer in range(self.layer_to_repair + 1, len(self.architecture)):
            ub = np.zeros((layer_values[0].shape[0], self.architecture[layer]))
            lb = np.zeros((layer_values[0].shape[0], self.architecture[layer]))
            for node_next in range(self.architecture[layer]):
                for s in range(layer_values[0].shape[0]):
                    lb[s, node_next] = bias[layer - 1][node_next]
                    ub[s, node_next] = bias[layer - 1][node_next]
                    for node_prev in range(self.architecture[layer - 1]):
                        w_temp = weights[layer - 1][node_prev][node_next]
                        lb[s, node_next] += lb_mat[
                            layer - self.layer_to_repair - 1
                        ][s][node_prev] * max(w_temp, 0) + ub_mat[
                            layer - self.layer_to_repair - 1
                        ][
                            s
                        ][
                            node_prev
                        ] * min(
                            w_temp, 0
                        )
                        ub[s, node_next] += ub_mat[
                            layer - self.layer_to_repair - 1
                        ][s][node_prev] * max(w_temp, 0) + lb_mat[
                            layer - self.layer_to_repair - 1
                        ][
                            s
                        ][
                            node_prev
                        ] * min(
                            w_temp, 0
                        )
                    # update stats
                    bound_stat_tracker.update_stats(
                        lb[s, node_next],
                        ub[s, node_next],
                        layer,
                        node_next,
                    )
            # print stats
            bound_stat_tracker.print_stats(layer)
            ub_mat.append(ub)
            lb_mat.append(lb)

        return ub_mat, lb_mat

    def __tight_bounds_lp(
        self, layer_values, weights, bias, ub_mat, lb_mat, max_weight_bound
    ):
        # initialize stat recorder
        bound_stat_tracker = BoundStatTracker(self.architecture)
        not_repair_list = [
            i
            for i in range(self.architecture[self.layer_to_repair])
            if i not in self.repair_node_list
        ]
        for l in range(self.layer_to_repair + 1, len(self.architecture)):
            if l == self.layer_to_repair + 1:
                next_node_list = self.repair_node_list
            else:
                next_node_list = range(self.architecture[l])
            for n in next_node_list:
                lp_model_layer = LPNNModel(
                    self.layer_to_repair,
                    self.architecture,
                    weights,
                    bias,
                    self.repair_node_list,
                    max_weight_bound,
                    self.param_precision,
                    self.data_precision,
                )
                x = lp_model_layer(l, n)
                for s in range(self.num_samples):
                    par_dict = {}
                    par_dict["inp"] = {
                        i: layer_values[self.layer_to_repair - 1][s][i]
                        if layer_values[self.layer_to_repair - 1][s][i] > 0
                        else 0.0
                        for i in range(
                            layer_values[self.layer_to_repair - 1][s].shape[0]
                        )
                    }
                    # bound_idx = 0
                    if len(not_repair_list) != 0:
                        par_dict[
                            "x" + str(self.layer_to_repair + 1) + "_param"
                        ] = {
                            i: layer_values[self.layer_to_repair][s][i]
                            for i in not_repair_list
                        }
                    for lay in range(self.layer_to_repair, l + 1):
                        par_dict[f"lb{lay+1}"] = {
                            i: lb_mat[lay - self.layer_to_repair][s][i]
                            for i in range(self.architecture[lay])
                        }
                        par_dict[f"ub{lay+1}"] = {
                            i: ub_mat[lay - self.layer_to_repair][s][i]
                            for i in range(self.architecture[lay])
                        }
                        # bound_idx += 1
                    lp_instance = lp_model_layer.model.create_instance(
                        {None: par_dict}
                    )

                    # minimum bound
                    cost_expr = getattr(lp_instance, x.name)[0]
                    lp_instance.obj = pyo.Objective(expr=cost_expr)
                    opt = pyo.SolverFactory("gurobi", solver_io="python")
                    opt.solve(lp_instance, tee=False)
                    lb_mat[l - self.layer_to_repair][s][n] = np.round(
                        getattr(lp_instance, x.name)[0]._value,
                        self.data_precision,
                    )
                    lp_instance.del_component("obj")

                    # maximum bound
                    cost_expr = -getattr(lp_instance, x.name)[0]
                    lp_instance.obj = pyo.Objective(expr=cost_expr)
                    opt.solve(lp_instance, tee=False)
                    ub_mat[l - self.layer_to_repair][s][n] = np.round(
                        getattr(lp_instance, x.name)[0]._value,
                        self.data_precision,
                    )
                    lp_instance.del_component("obj")
                    # update stats
                    bound_stat_tracker.update_stats(
                        lb_mat[l - self.layer_to_repair][s][n],
                        ub_mat[l - self.layer_to_repair][s][n],
                        l,
                        n,
                    )
                # print stats
            bound_stat_tracker.print_stats(l)

        return ub_mat, lb_mat

    def __set_param_n_output_bounds(
        self,
        param_bounds,
        output_bounds,
        weights,
        bias,
        max_weight_bound,
        ub_mat,
        lb_mat,
    ) -> Any:

        """_summary_

        Args:
            param_bounds:
            output_bounds:
            weights:
            bias:
            max_weight_bound:

        Returns:
            _type_: _description_
        """

        if param_bounds is None:
            param_bounds = (
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
            )

        if output_bounds is None:
            #######################
            # TODO: (12_7_2022) the global upper and lower node bounds
            # specified by lb and ub
            ub = 0.0
            for _, item in enumerate(ub_mat):
                if np.max(item) > ub:
                    ub = np.max(item)
            lb = 0.0
            for _, item in enumerate(lb_mat):
                if np.min(item) < lb:
                    lb = np.min(item)
            output_bounds = (lb, ub)

        return param_bounds, output_bounds

    def __solve_optimization_problem(
        self,
        gdp_formulation: str,
        solver_factory: str,
        optimizer_options: dict,
    ) -> None:

        """Solve the Optimization Problem

        Args:
            gdp_formulation: Formulation of GDP
            solver_factory: Solver Factory
            optimizer_options: Optimizer Options Dictionary

        Raises:
            ValueError: "Optimization model does not exist. First compile the model!"
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
        if hasattr(self.opt_model, "db"):
            print("----------------------")
            print(self.opt_model.db.display())

    def __give_mlp_model(self, model: Any, model_type: str) -> Any:

        if model_type == "tensorflow":
            architecture = tf2_get_architecture(model)
            model_mlp = MLP(
                architecture[0],
                architecture[-1],
                architecture[1:-1],
            )
            model_mlp.set_mlp_params(tf2_get_weights(model))
        elif model_type == "pytorch":
            architecture = pt_get_architecture(model)
            model_mlp = MLP(
                architecture[0],
                architecture[-1],
                architecture[1:-1],
            )
            model_mlp.set_mlp_params(pt_get_weights(model))
        else:
            raise TypeError(
                f"Expected tensorflow or pytorch. Received {model_type} instead."
            )

        return model_mlp, architecture
