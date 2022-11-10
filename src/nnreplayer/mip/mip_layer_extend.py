import pyomo.environ as pyo
import pyomo.gdp as pyg
from pyomo.gdp import *
from nnreplayer.utils import generate_output_constraints
from nnreplayer.utils import ConstraintsClass
import numpy as np
import numpy.typing as npt
from typing import List, Union


class MIPLayerExtend:
    """_summary_"""

    def __init__(
        self,
        model,
        last_ctrl_layer: int,
        uin: int,
        uout: int,
        weights: npt.NDArray,
        bias: npt.NDArray,
        repair_node_list: List[int],
        w_error_norm: int = 0,
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
        ############################################
        # TODO: Speratare activated and deactivated nodes
        self.w_error_norm = w_error_norm
        self.repair_node_list = repair_node_list  # list of repair nodes
        # detect if this layer is a repair layer and if so,
        # which nodes are repaired. If this layer is not a repair layer,
        # then its weight and bias terms are fixed to the original values.
        self.w = weights
        self.b = bias
        model.nlayers += 1
        self.model = model
        self.last_ctrl_layer = last_ctrl_layer
        ############################################

    def __call__(
        self,
        x: npt.NDArray,
        shape,
        output_constraint_list: List[ConstraintsClass],
        ##############################
        # TODO: (12_7_2022) input the upper and lower bounds of nodes for
        # each sample
        ub: npt.NDArray,
        lb: npt.NDArray,
        ##############################
        relu: bool = False,
        # output_bounds: tuple = (-1e1, 1e1),
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
            # define relu layer constraints
            ##############################
            # TODO: (12_7_2022) input the upper and lower bounds of nodes for
            # each sample
            x_next = self._relu_constraints(x, shape, ub, lb)
            ##############################
        else:
            # define linear layer constraints
            ##############################
            # TODO: (12_7_2022) input the upper and lower bounds of nodes for
            # each sample
            x_next = self._constraints(
                x,
                shape,
                # self.layer_num_next,
                output_constraint_list,
                ub,
                lb,
            )
            ##############################

        return x_next

    def _relu_constraints(
        self,
        x: npt.NDArray,
        shape: tuple,
        # l,
        ##############################
        # TODO: (12_7_2022) input the upper and lower bounds of nodes for
        # each sample
        ub: npt.NDArray,
        lb: npt.NDArray,
        ##############################
        # max_weight_bound: Union[int, float] = 1,
        # output_bounds: tuple = (-1e1, 1e1),
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
        next_node_list = list(range(self.uout))

        # define the variables of the next nodes
        def x_next_bound(model, i, j):
            return (0.0, ub[i, j])

        setattr(
            self.model,
            x_l,
            pyo.Var(
                range(m),
                next_node_list,
                domain=pyo.NonNegativeReals,
                bounds=x_next_bound,
            ),
        )

        setattr(
            self.model,
            theta_l,
            pyo.Var(range(m), next_node_list, domain=pyo.Binary),
        )

        # Big-M method constraints
        # inequality x_l >= w^Tx+b
        def constraint_1(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            # return constraint based on the activation status of the node
            if lb[i, j] >= 0:
                return product == getattr(model, x_l)[i, j]
            elif ub[i, j] <= 0:
                return getattr(model, x_l)[i, j] == 0
            else:
                return product <= getattr(model, x_l)[i, j]

        setattr(
            self.model,
            "constraint_inequality_1_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(m),
                next_node_list,
                rule=constraint_1,
            ),
        )

        # inequality x_l <= w^Tx+b - (1-theta)*LB
        def constraint_2(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            # return constraint based on the activation status of the node
            if lb[i, j] >= 0:
                return pyo.Constraint.Skip
            elif ub[i, j] <= 0:
                return pyo.Constraint.Skip
            else:
                return (
                    product - lb[i, j] * (1 - getattr(model, theta_l)[i, j])
                    >= getattr(model, x_l)[i, j]
                )

        setattr(
            self.model,
            "constraint_inequality_2_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(m),
                next_node_list,
                rule=constraint_2,
            ),
        )

        # inequality x_l <= theta*UB
        def constraint_3(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            # return constraint based on the activation status of the node
            if lb[i, j] >= 0:
                return pyo.Constraint.Skip
            elif ub[i, j] <= 0:
                return pyo.Constraint.Skip
            else:
                return (
                    ub[i, j] * getattr(model, theta_l)[i, j]
                    >= getattr(model, x_l)[i, j]
                )

        setattr(
            self.model,
            "constraint_inequality_3_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(m),
                next_node_list,
                rule=constraint_3,
            ),
        )

        # collect the values of next layer
        x_next = []  # values of next layer
        for s in range(m):
            x_temp = []
            for c in range(self.uout):
                x_temp.append(getattr(self.model, x_l)[s, c])
            x_next.append(x_temp)

        return x_next
        ###############################################################

    def _constraints(
        self,
        x: npt.NDArray,
        shape: tuple,
        # l,
        output_constraint_list: List[ConstraintsClass],
        ##############################
        # TODO: (12_7_2022) input the upper and lower bounds of nodes for
        # each sample
        ub: npt.NDArray,
        lb: npt.NDArray,
        ##############################
        # max_weight_bound: Union[int, float] = 10,
        # output_bounds: tuple = (-1e1, 1e1),
    ):
        """_summary_

        Args:
            x (npt.NDArray): _description_
            shape (tuple): _description_
            l (_type_): _description_
            output_constraint_list (List[ConstraintsClass]): _description_
            max_weight_bound (Union[int, float], optional): _description_. Defaults to 10.
            output_bounds (tuple, optional): _description_. Defaults to (-1e1, 1e1).

        Returns:
            _type_: _description_
        """

        m, n = shape
        assert n == self.uin

        x_l = "x" + str(self.layer_num_next)

        def x_next_bound(model, i, j):
            return (lb[i, j], ub[i, j])

        # x_l = 'x'+str(self.layer_num_next)
        setattr(
            self.model,
            x_l,
            pyo.Var(
                range(m),
                range(self.uout),
                domain=pyo.Reals,
                bounds=x_next_bound,
            ),
        )

        def constraints(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            return product == getattr(model, x_l)[i, j]

        setattr(
            self.model,
            "eq_constraint" + str(self.layer_num_next),
            pyo.Constraint(range(m), range(self.uout), rule=constraints),
        )

        if output_constraint_list:
            constraint_addition_string = generate_output_constraints(
                output_constraint_list
            )
            exec(constraint_addition_string, locals(), globals())

        return getattr(self.model, x_l)
