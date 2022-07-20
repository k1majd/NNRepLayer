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
        max_weight_bound: Union[int, float] = 1.0,
        param_precision: int = 6,
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
        self.not_repair_node_list = [
            i for i in range(self.uout) if i not in self.repair_node_list
        ]
        # detect if this layer is a repair layer and if so,
        # which nodes are repaired. If this layer is not a repair layer,
        # then its weight and bias terms are fixed to the original values.
        if model.nlayers == layer_to_repair and num_layers_ahead != 0:

            def w_bound(model, i, j):
                return (
                    np.round(
                        weights[i, repair_node_list[j]] - max_weight_bound,
                        param_precision,
                    ),
                    np.round(
                        weights[i, repair_node_list[j]] + max_weight_bound,
                        param_precision,
                    ),
                )

            def w_initialize(model, i, j):
                return weights[i, repair_node_list[j]]

            def b_bound(model, j):
                return (
                    np.round(
                        bias[repair_node_list[j]] - max_weight_bound,
                        param_precision,
                    ),
                    np.round(
                        bias[repair_node_list[j]] + max_weight_bound,
                        param_precision,
                    ),
                )

            def b_initialize(model, j):
                return bias[repair_node_list[j]]

            w_l, b_l = "w_r", "b_r"
            setattr(
                model,
                w_l,
                pyo.Var(
                    range(uin),
                    range(len(repair_node_list)),
                    domain=pyo.Reals,
                    bounds=w_bound,
                    initialize=w_initialize,
                ),
            )
            setattr(
                model,
                b_l,
                pyo.Var(
                    range(len(repair_node_list)),
                    domain=pyo.Reals,
                    bounds=b_bound,
                    initialize=b_initialize,
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
        # shape,
        # ub: npt.NDArray,
        # lb: npt.NDArray,
        final_layer: int,
        final_node: int,
        x: npt.NDArray = np.array([0.0]),
        # relu: bool = False,
        # max_weight_bound: Union[int, float] = 10,
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
        if self.layer_num == self.layer_to_repair:
            x_next = self._relu_constraints_firts_layer()
        else:
            x_next = self._relu_constraints(
                x,
                final_layer,
                final_node,
                # max_weight_bound,
                # output_bounds,
            )
        return x_next

    def _relu_constraints_firts_layer(
        self,
        # x: npt.NDArray,
        # shape: tuple,
        # ub: npt.NDArray,
        # lb: npt.NDArray,
        # max_weight_bound: Union[int, float] = 1.0,
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
        # define model input as parameter
        num_next_repair_nodes = len(self.repair_node_list)
        self.model.inp = pyo.Param(range(self.uin))
        # self.model.x_inactive_next = pyo.Param(
        #     range(self.uout - num_next_repair_nodes)
        # )
        x_l, theta_l, ub_l, lb_l = (
            "x" + str(self.layer_num_next),
            "theta" + str(self.layer_num_next),
            "ub" + str(self.layer_num_next),
            "lb" + str(self.layer_num_next),
        )
        setattr(self.model, ub_l, pyo.Param(range(self.uout)))
        setattr(self.model, lb_l, pyo.Param(range(self.uout)))

        # define the variables of the next nodes
        def x_next_bound(model, j):
            return (0.0, getattr(model, ub_l)[self.repair_node_list[j]])

        setattr(
            self.model,
            x_l,
            pyo.Var(
                range(num_next_repair_nodes),
                domain=pyo.NonNegativeReals,
                bounds=x_next_bound,
            ),
        )
        # relaxed theta \in [0,1]
        setattr(
            self.model,
            theta_l,
            pyo.Var(
                range(num_next_repair_nodes),
                domain=pyo.NonNegativeReals,
                bounds=(0.0, 1.0),
            ),
        )

        # Big-M method constraints
        # inequality x_l >= w^Tx+b
        def constraint_1(model, j):
            product = model.b_r[j]
            for k in range(self.uin):
                product += model.inp[k] * model.w_r[k, j]
            # return constraint based on the activation status of the node
            if getattr(model, lb_l)[self.repair_node_list[j]] >= 0:
                return product == getattr(model, x_l)[j]
            elif getattr(model, ub_l)[self.repair_node_list[j]] <= 0:
                return getattr(model, x_l)[j] == 0
            else:
                return product <= getattr(model, x_l)[j]

        setattr(
            self.model,
            "constraint_inequality_1_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(num_next_repair_nodes),
                rule=constraint_1,
            ),
        )

        # inequality x_l <= w^Tx+b - (1-theta)*LB
        def constraint_2(model, j):
            product = model.b_r[j]
            for k in range(self.uin):
                product += model.inp[k] * model.w_r[k, j]
            # return constraint based on the activation status of the node
            if getattr(model, lb_l)[self.repair_node_list[j]] >= 0:
                return pyo.Constraint.Skip
            elif getattr(model, ub_l)[self.repair_node_list[j]] <= 0:
                return pyo.Constraint.Skip
            else:
                return (
                    product
                    - getattr(model, lb_l)[self.repair_node_list[j]]
                    * (1 - getattr(model, theta_l)[j])
                    >= getattr(model, x_l)[j]
                )

        setattr(
            self.model,
            "constraint_inequality_2_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(num_next_repair_nodes),
                rule=constraint_2,
            ),
        )

        # inequality x_l <= theta*UB
        def constraint_3(model, j):
            product = model.b_r[j]
            for k in range(self.uin):
                product += model.inp[k] * model.w_r[k, j]
            # return constraint based on the activation status of the node
            if getattr(model, lb_l)[self.repair_node_list[j]] >= 0:
                return pyo.Constraint.Skip
            elif getattr(model, ub_l)[self.repair_node_list[j]] <= 0:
                return pyo.Constraint.Skip
            else:
                return (
                    getattr(model, ub_l)[self.repair_node_list[j]]
                    * getattr(model, theta_l)[j]
                    >= getattr(model, x_l)[j]
                )

        setattr(
            self.model,
            "constraint_inequality_3_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(num_next_repair_nodes),
                rule=constraint_3,
            ),
        )

        return getattr(self.model, x_l)
        # collect the values of next layer
        # x_next = []  # values of next layer
        # for c in range(self.uout):
        #     if c in self.repair_node_list:
        #         x_next.append(
        #             getattr(self.model, x_l)[self.repair_node_list.index(c)]
        #         )
        #     else:
        #         # ||||||||||
        #         # ----> you may need to define this part with pyo.Param
        #         # x_val_temp = self.b_orig[c]
        #         # for k in range(self.uin):
        #         #     x_val_temp += self.model.inp[k] * self.w_orig[k, c]
        #         # # x_val_temp = pyo.maximize(0.,x_val_temp)
        #         # x_next.append(
        #         #     x_val_temp
        #         # ) if x_val_temp >= 0.0 else x_next.append(0.0)
        #         x_next.append(
        #             self.model.x_inactive_next[
        #                 self.not_repair_node_list.index(c)
        #             ]
        #         )

        # return x_next

    def _relu_constraints(
        self,
        x: npt.NDArray,
        # shape: tuple,
        # ub: npt.NDArray,
        # lb: npt.NDArray,
        final_layer: int,
        final_node: int,
        # max_weight_bound: Union[int, float] = 1.0,
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
        # Detect final layer
        ub_l, lb_l = (
            "ub" + str(self.layer_num_next),
            "lb" + str(self.layer_num_next),
        )
        setattr(self.model, ub_l, pyo.Param(range(self.uout)))
        setattr(self.model, lb_l, pyo.Param(range(self.uout)))
        if self.layer_num == final_layer:
            num_next_repair_nodes = 1

            def x_next_bound(model, j):
                return (
                    getattr(model, lb_l)[final_node],
                    getattr(model, ub_l)[final_node],
                )

            # define the variables of the next nodes
            x_l = "x" + str(self.layer_num_next)
            setattr(
                self.model,
                x_l,
                pyo.Var(
                    range(num_next_repair_nodes),
                    domain=pyo.Reals,
                    bounds=x_next_bound,
                ),
            )

            # Big-M method constraints
            # inequality x_l >= w^Tx+b
            def constraint_1(model, j):
                product = self.b[j]
                for k in range(self.uin):
                    if x.name.split("x")[1] == "1":
                        product += model.x1[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "2":
                        product += model.x2[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "3":
                        product += model.x3[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "4":
                        product += model.x4[k] * self.w[k, j]
                    # product += x[k] * self.w[k, j]
                # return constraint based on the activation status of the node
                if getattr(model, ub_l)[j] <= 0:
                    return getattr(model, x_l)[j] == 0
                else:
                    return product == getattr(model, x_l)[j]

            setattr(
                self.model,
                "constraint_inequality_1_lay" + str(self.layer_num_next),
                pyo.Constraint(
                    range(num_next_repair_nodes),
                    rule=constraint_1,
                ),
            )

        else:
            num_next_repair_nodes = self.uout

            def x_next_bound(model, j):
                return (0.0, getattr(model, ub_l)[j])

            # define the variables of the next nodes
            x_l, theta_l = (
                "x" + str(self.layer_num_next),
                "theta" + str(self.layer_num_next),
            )
            setattr(
                self.model,
                x_l,
                pyo.Var(
                    range(num_next_repair_nodes),
                    domain=pyo.NonNegativeReals,
                    bounds=x_next_bound,
                ),
            )
            setattr(
                self.model,
                theta_l,
                pyo.Var(
                    range(num_next_repair_nodes),
                    domain=pyo.NonNegativeReals,
                    bounds=(0.0, 1.0),
                ),
            )

            # Big-M method constraints
            # inequality x_l >= w^Tx+b
            def constraint_1(model, j):
                product = self.b[j]
                for k in range(self.uin):
                    if x.name.split("x")[1] == "1":
                        product += model.x1[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "2":
                        product += model.x2[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "3":
                        product += model.x3[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "4":
                        product += model.x4[k] * self.w[k, j]
                # return constraint based on the activation status of the node
                if getattr(model, lb_l)[j] >= 0:
                    return product == getattr(model, x_l)[j]
                elif getattr(model, ub_l)[j] <= 0:
                    return getattr(model, x_l)[j] == 0
                else:
                    return product <= getattr(model, x_l)[j]

            setattr(
                self.model,
                "constraint_inequality_1_lay" + str(self.layer_num_next),
                pyo.Constraint(
                    range(num_next_repair_nodes),
                    rule=constraint_1,
                ),
            )

            # inequality x_l <= w^Tx+b - (1-theta)*LB
            def constraint_2(model, j):
                product = self.b[j]
                for k in range(self.uin):
                    if x.name.split("x")[1] == "1":
                        product += model.x1[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "2":
                        product += model.x2[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "3":
                        product += model.x3[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "4":
                        product += model.x4[k] * self.w[k, j]
                # return constraint based on the activation status of the node
                if getattr(model, lb_l)[j] >= 0:
                    return pyo.Constraint.Skip
                elif getattr(model, ub_l)[j] <= 0:
                    return pyo.Constraint.Skip
                else:
                    return (
                        product
                        - getattr(model, lb_l)[j]
                        * (1 - getattr(model, theta_l)[j])
                        >= getattr(model, x_l)[j]
                    )

            setattr(
                self.model,
                "constraint_inequality_2_lay" + str(self.layer_num_next),
                pyo.Constraint(
                    range(num_next_repair_nodes),
                    rule=constraint_2,
                ),
            )

            # inequality x_l <= theta*UB
            def constraint_3(model, j):
                product = self.b[j]
                for k in range(self.uin):
                    if x.name.split("x")[1] == "1":
                        product += model.x1[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "2":
                        product += model.x2[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "3":
                        product += model.x3[k] * self.w[k, j]
                    elif x.name.split("x")[1] == "4":
                        product += model.x4[k] * self.w[k, j]
                # return constraint based on the activation status of the node
                if getattr(model, lb_l)[j] >= 0:
                    return pyo.Constraint.Skip
                elif getattr(model, ub_l)[j] <= 0:
                    return pyo.Constraint.Skip
                else:
                    return (
                        getattr(model, ub_l)[j] * getattr(model, theta_l)[j]
                        >= getattr(model, x_l)[j]
                    )

            setattr(
                self.model,
                "constraint_inequality_3_lay" + str(self.layer_num_next),
                pyo.Constraint(
                    range(num_next_repair_nodes),
                    rule=constraint_3,
                ),
            )

        return getattr(self.model, x_l)
        # collect the values of next layer
        # x_next = []  # values of next layer
        # for c in range(num_next_repair_nodes):
        #     x_next.append(getattr(self.model, x_l)[c])

        # return x_next
        ###############################################################
