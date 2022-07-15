import pyomo.environ as pyo
from pyomo.gdp import *
from nnreplayer.utils import generate_output_constraints
from nnreplayer.utils import ConstraintsClass
import numpy as np
import numpy.typing as npt
from typing import List, Union


class LPLayer:
    """_summary_"""

    def __init__(
        self,
        model,
        layer_to_repair: int,
        uin: int,
        uout: int,
        weights: npt.NDArray,
        bias: npt.NDArray,
        num_layers_ahead: int,
        repair_node_list: List[int],
        param_bounds: tuple = (-1e4, 1e4),
    ) -> None:
        """Initialize the MIPLayer class.
           Specify the weight and bias terms of the given layer.
           Non-repair layers and nodes are fixed to the original weight and bias terms.
           OPT variables for repair nodes are created.

        Args:
            model (_type_): _description_
            layer_to_repair (int): _description_
            uin (int): _description_
            uout (int): _description_
            weights (npt.NDArray): _description_
            bias (npt.NDArray): _description_
            num_layers_ahead (int): _description_
            repair_node_list (List[int]): _description_
            w_error_norm (int, optional): weight error norm type 0 = L-inf, 1 = L-1. Defaults to 0.
            param_bounds (tuple, optional): _description_. Defaults to (-1, 1).
        """

        model.nlayers = getattr(model, "nlayers", 0)
        self.layer_num = model.nlayers
        self.uin, self.uout = uin, uout
        self.repair_node_list = repair_node_list  # list of repair nodes
        # detect if this layer is a repair layer and if so,
        # which nodes are repaired. If this layer is not a repair layer,
        # then its weight and bias terms are fixed to the original values.
        if model.nlayers == layer_to_repair and num_layers_ahead != 0:
            w_l, b_l = "w" + str(model.nlayers), "b" + str(model.nlayers)
            setattr(
                model,
                w_l,
                pyo.Var(
                    range(uin),
                    range(len(repair_node_list)),
                    domain=pyo.Reals,
                    bounds=param_bounds,
                ),
            )
            setattr(
                model,
                b_l,
                pyo.Var(
                    range(len(repair_node_list)),
                    domain=pyo.Reals,
                    bounds=param_bounds,
                ),
            )
            self.w = getattr(model, w_l)
            self.b = getattr(model, b_l)

            self.w_orig = weights
            self.b_orig = bias
        else:
            self.w = weights
            self.b = bias

        model.nlayers += 1
        self.model = model
        self.layer_to_repair = layer_to_repair

    def __call__(
        self,
        x: npt.NDArray,
        shape,
        ub: npt.NDArray,
        lb: npt.NDArray,
        relu: bool = False,
        max_weight_bound: Union[int, float] = 10,
        output_bounds: tuple = (-1e1, 1e1),
    ):
        """_summary_

        Args:
            x: _description_
            shape: _description_
            output_constraint_list: _description_
            ub: nodes upper bound for each sample
            lb: nodes lower bound for each sample
            relu: _description_. Defaults to False.
            max_weight_bound (optional): _description_. Defaults to 10.
            output_bounds (optional): _description_. Defaults to (-1e1, 1e1).

        Returns:
            _type_: _description_
        """

        self.layer_num_next = (
            getattr(self, "layer_num", 0) + 1
        )  # next layer number
        if relu:
            x_next = self._relu_constraints(
                x, shape, ub, lb, max_weight_bound, output_bounds
            )

        # define w bounding constraints
        if self.layer_num == self.layer_to_repair:
            print(f"Repairing {self.layer_to_repair}th layer")
            if self.w_error_norm == 0:  # l-inf norm error
                self._weight_bound_constraint_linf(max_weight_bound)
            else:  # l-1 norm error
                self._weight_bound_constraint_l1_new(max_weight_bound)
        return x_next

    def _relu_constraints(
        self,
        x: npt.NDArray,
        shape: tuple,
        ub: npt.NDArray,
        lb: npt.NDArray,
        max_weight_bound: Union[int, float] = 1,
        output_bounds: tuple = (-1e1, 1e1),
    ):
        """_summary_

        Args:
            x (npt.NDArray): _description_
            shape (tuple): _description_
            l (_type_): _description_
            max_weight_bound (Union[int, float], optional): _description_. Defaults to 1.
            output_bounds (tuple, optional): _description_. Defaults to (-1e1, 1e1).

        Returns:
            _type_: _description_
        """
        m, n = shape
        assert n == self.uin

        x_l, theta_l = (
            "x" + str(self.layer_num_next),
            "theta" + str(self.layer_num_next),
        )

        ###############################################################
        # TODO: Edit this part
        ## check if this layer is the repair layer
        if self.layer_num == self.layer_to_repair:
            num_next_repair_nodes = len(self.repair_node_list)
        else:
            num_next_repair_nodes = self.uout

        # define the variables of the next nodes
        setattr(
            self.model,
            x_l,
            pyo.Var(
                range(m),
                range(num_next_repair_nodes),
                domain=pyo.NonNegativeReals,
                bounds=output_bounds,
            ),
        )
        setattr(
            self.model,
            theta_l,
            pyo.Var(range(m), range(num_next_repair_nodes), domain=pyo.Binary),
        )

        # Big-M method constraints
        # inequality x_l >= w^Tx+b
        def constraint_1(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            # return constraint based on the activation status of the node
            if lb[i, self.repair_node_list[j]] >= 0:
                return product == getattr(model, x_l)[i, j]
            elif ub[i, self.repair_node_list[j]] <= 0:
                return getattr(model, x_l)[i, j] == 0
            else:
                return product <= getattr(model, x_l)[i, j]

        setattr(
            self.model,
            "constraint_inequality_1_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(m),
                range(num_next_repair_nodes),
                rule=constraint_1,
            ),
        )

        # inequality x_l <= w^Tx+b - (1-theta)*LB
        def constraint_2(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            # return constraint based on the activation status of the node
            if lb[i, self.repair_node_list[j]] >= 0:
                return pyo.Constraint.Skip
            elif ub[i, self.repair_node_list[j]] <= 0:
                return pyo.Constraint.Skip
            else:
                return (
                    product
                    - lb[i, self.repair_node_list[j]]
                    * (1 - getattr(model, theta_l)[i, j])
                    >= getattr(model, x_l)[i, j]
                )

        setattr(
            self.model,
            "constraint_inequality_2_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(m),
                range(num_next_repair_nodes),
                rule=constraint_2,
            ),
        )

        # inequality x_l <= theta*UB
        def constraint_3(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            # return constraint based on the activation status of the node
            if lb[i, self.repair_node_list[j]] >= 0:
                return pyo.Constraint.Skip
            elif ub[i, self.repair_node_list[j]] <= 0:
                return pyo.Constraint.Skip
            else:
                return (
                    ub[i, self.repair_node_list[j]]
                    * getattr(model, theta_l)[i, j]
                    >= getattr(model, x_l)[i, j]
                )

        setattr(
            self.model,
            "constraint_inequality_3_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(m),
                range(num_next_repair_nodes),
                rule=constraint_3,
            ),
        )

        # collect the values of next layer
        x_next = []  # values of next layer
        if self.layer_num == self.layer_to_repair:
            for s in range(m):
                x_temp = []
                for c in range(self.uout):
                    if c in self.repair_node_list:
                        x_temp.append(
                            getattr(self.model, x_l)[
                                s, self.repair_node_list.index(c)
                            ]
                        )
                    else:
                        x_val_temp = self.b_orig[c]
                        for k in range(self.uin):
                            x_val_temp += x[s][k] * self.w_orig[k, c]
                        x_temp.append(
                            x_val_temp
                        ) if x_val_temp >= 0.0 else x_temp.append(0.0)
                x_next.append(x_temp)
        else:
            for s in range(m):
                x_temp = []
                for c in range(self.uout):
                    x_temp.append(getattr(self.model, x_l)[s, c])
                x_next.append(x_temp)

        return x_next
        ###############################################################
