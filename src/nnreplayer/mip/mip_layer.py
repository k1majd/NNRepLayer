import pyomo.environ as pyo
import pyomo.gdp as pyg
from pyomo.gdp import *
from nnreplayer.utils import generate_output_constraints
from nnreplayer.utils import ConstraintsClass
import numpy as np
import numpy.typing as npt
from typing import List, Union


class MIPLayer:
    """_summary_"""

    def __init__(
        self,
        model,
        layer_to_repair: int,
        uin: int,
        uout: int,
        weights: npt.NDArray,
        bias: npt.NDArray,
        ####################################
        # TODO: add these parameters
        num_layers_ahead: int,
        repair_node_list: List[int],
        # weight_activations: npt.NDArray,
        # bias_activations: npt.NDArray,
        # max_weight_bound: Union[int, float],
        ####################################
        param_bounds: tuple = (-1, 1),
    ) -> None:
        """_summary_

        Args:
            model (_type_): _description_
            layer_to_repair (int): _description_
            uin (int): _description_
            uout (int): _description_
            weights (npt.NDArray): _description_
            bias (npt.NDArray): _description_
            param_bounds (tuple, optional): _description_. Defaults to (-1, 1).
        """

        model.nlayers = getattr(model, "nlayers", 0)
        # print("Model layers = {}".format(model.nlayers))
        self.layer_num = model.nlayers
        self.uin, self.uout = uin, uout
        ############################################
        # TODO: Speratare activated and deactivated weight and bias terms
        # self.label = model.nlayers  # label of this layer
        self.repair_node_list = repair_node_list  # list of repair nodes
        # specify repair weights in the repair layers
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

            # TODO: Remove this part
            # w_l, b_l = "w" + str(model.nlayers), "b" + str(model.nlayers)

            # setattr(
            #     model,
            #     w_l,
            #     pyo.Var(
            #         range(uin),
            #         range(uout),
            #         domain=pyo.Reals,
            #         bounds=param_bounds,
            #     ),
            # )
            # setattr(
            #     model,
            #     b_l,
            #     pyo.Var(range(uout), domain=pyo.Reals, bounds=param_bounds),
            # )

            # self.w = getattr(model, w_l)
            # self.b = getattr(model, b_l)

            self.w_orig = weights
            self.b_orig = bias
        # ensure that all nodes of last layer are repaired
        # even if this layer is the target repair layer
        elif model.nlayers == layer_to_repair and num_layers_ahead == 0:
            w_l, b_l = "w" + str(model.nlayers), "b" + str(model.nlayers)
            setattr(
                model,
                w_l,
                pyo.Var(
                    range(uin),
                    range(uout),
                    domain=pyo.Reals,
                    bounds=param_bounds,
                ),
            )
            setattr(
                model,
                b_l,
                pyo.Var(range(uout), domain=pyo.Reals, bounds=param_bounds),
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
        ############################################

    def __call__(
        self,
        x: npt.NDArray,
        shape,
        output_constraint_list: List[ConstraintsClass],
        relu: bool = False,
        max_weight_bound: Union[int, float] = 10,
        output_bounds: tuple = (-1e1, 1e1),
    ):
        """_summary_

        Args:
            x (_type_): _description_
            shape (_type_): _description_
            output_constraint_list (_type_): _description_
            relu (bool, optional): _description_. Defaults to False.
            max_weight_bound (int, optional): _description_. Defaults to 10.
            output_bounds (tuple, optional): _description_. Defaults to (-1e1, 1e1).

        Returns:
            _type_: _description_
        """

        self.lout = getattr(self, "layer_num", 0) + 1
        if relu:
            return self._relu_constraints(
                x, shape, self.lout, max_weight_bound, output_bounds
            )
        return self._constraints(
            x,
            shape,
            self.lout,
            output_constraint_list,
            max_weight_bound,
            output_bounds,
        )

    def _relu_constraints(
        self,
        x: npt.NDArray,
        shape: tuple,
        l,
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

        x_l, s_l, theta_l = "x" + str(l), "s" + str(l), "theta" + str(l)

        ###############################################################
        # TODO: Edit this part
        ## check if this layer is the repair layer
        if l == self.layer_to_repair + 1:
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
            s_l,
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

        # add the linear constraints related to the target repair nodes
        def constraints(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            return (
                product
                == getattr(model, x_l)[i, j] - getattr(model, s_l)[i, j]
            )

        setattr(
            self.model,
            "eq_constraint" + str(l),
            pyo.Constraint(
                range(m), range(num_next_repair_nodes), rule=constraints
            ),
        )

        # add the integer constraints related to the target repair nodes
        def disjuncts(model, i, j):
            return [
                (
                    getattr(model, theta_l)[i, j] == 0,
                    getattr(model, x_l)[i, j] <= 0,
                ),
                (
                    getattr(model, theta_l)[i, j] == 1,
                    getattr(model, s_l)[i, j] <= 0,
                ),
            ]

        setattr(
            self.model,
            "disjunction" + str(l),
            pyg.Disjunction(
                range(m), range(num_next_repair_nodes), rule=disjuncts
            ),
        )

        # collect the values of next layer
        x_next = []  # values of next layer
        if l == self.layer_to_repair + 1:
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

        if l == self.layer_to_repair + 1:
            print("Activating mid layer")
            w_l = "w" + str(l - 1)
            b_l = "b" + str(l - 1)

            dw_l = "dw"
            setattr(
                self.model,
                dw_l,
                pyo.Var(
                    within=pyo.NonNegativeReals, bounds=(0, max_weight_bound)
                ),
            )

            def constraint_bound_w0(model, i, j):
                return getattr(model, w_l)[
                    i, self.repair_node_list.index(j)
                ] - self.w_orig[i, j] <= getattr(model, dw_l)

            def constraint_bound_w1(model, i, j):
                return getattr(model, w_l)[
                    i, self.repair_node_list.index(j)
                ] - self.w_orig[i, j] >= -getattr(model, dw_l)

            def constraint_bound_b0(model, j):
                return getattr(model, b_l)[
                    self.repair_node_list.index(j)
                ] - self.b_orig[j] <= getattr(model, dw_l)

            def constraint_bound_b1(model, j):
                return getattr(model, b_l)[
                    self.repair_node_list.index(j)
                ] - self.b_orig[j] >= -getattr(model, dw_l)

            setattr(
                self.model,
                "w_bounded_constraint0" + str(l),
                pyo.Constraint(
                    range(self.uin),
                    self.repair_node_list,
                    rule=constraint_bound_w0,
                ),
            )
            setattr(
                self.model,
                "w_bounded_constraint1" + str(l),
                pyo.Constraint(
                    range(self.uin),
                    self.repair_node_list,
                    rule=constraint_bound_w1,
                ),
            )
            setattr(
                self.model,
                "b_bounded_constraint0" + str(l),
                pyo.Constraint(
                    self.repair_node_list, rule=constraint_bound_b0
                ),
            )
            setattr(
                self.model,
                "b_bounded_constraint1" + str(l),
                pyo.Constraint(
                    self.repair_node_list, rule=constraint_bound_b1
                ),
            )

        return x_next
        ###############################################################

    def _constraints(
        self,
        x: npt.NDArray,
        shape: tuple,
        l,
        output_constraint_list: List[ConstraintsClass],
        max_weight_bound: Union[int, float] = 10,
        output_bounds: tuple = (-1e1, 1e1),
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

        x_l = "x" + str(l)

        # x_l = 'x'+str(l)
        setattr(
            self.model,
            x_l,
            pyo.Var(
                range(m),
                range(self.uout),
                domain=pyo.Reals,
                bounds=output_bounds,
            ),
        )

        def constraints(model, i, j):
            product = self.b[j]
            for k in range(self.uin):
                product += x[i][k] * self.w[k, j]
            return product == getattr(model, x_l)[i, j]

        setattr(
            self.model,
            "eq_constraint" + str(l),
            pyo.Constraint(range(m), range(self.uout), rule=constraints),
        )

        if output_constraint_list:
            constraint_addition_string = generate_output_constraints(
                output_constraint_list
            )
            exec(constraint_addition_string, locals(), globals())
        # def constraint_inside0(model, i):
        #     return [(getattr(model, ind_l)[i, 0] == 0, getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 1],
        #             getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 2],
        #             getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 3],
        #             getattr(model, x_l)[i, 0] + 0.0001 <= getattr(model, x_l)[i, 4]),
        #             (getattr(model, ind_l)[i, 0] == 1, getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 0],
        #             getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 2],
        #             getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 3],
        #             getattr(model, x_l)[i, 1] + 0.0001 <= getattr(model, x_l)[i, 4])]

        # setattr(self.model, 'keep_inside_constraint0'+str(l),
        #                 pyg.Disjunction(range(m), rule=constraint_inside0))

        # ########## Define the output constraint here ########################
        # def constraint_inside0(model, i):
        #     return [[getattr(model, x_l)[i, 0] - 0.45 <= 0],
        #             [0.55 - getattr(model, x_l)[i, 0] <= 0],
        #             [getattr(model, x_l)[i, 0] - 0.55 <= 0, 0.45 - getattr(model, x_l)[i, 0] <= 0, 0.25 - getattr(model, x_l)[i, 1] <= 0],
        #             [getattr(model, x_l)[i, 0] - 0.55 <= 0, 0.45 - getattr(model, x_l)[i, 0] <= 0, getattr(model, x_l)[i, 1] - 0.1 <= 0]]

        # setattr(self.model, 'keep_inside_constraint0'+str(l),
        #          pyg.Disjunction(range(m), rule=constraint_inside0))
        # #####################################################################

        ###############################################################
        # TODO: Edit this part
        if l == self.layer_to_repair + 1:
            print("Activating Last layer")
            w_l = "w" + str(l - 1)
            b_l = "b" + str(l - 1)

            dw_l = "dw"
            setattr(
                self.model,
                dw_l,
                pyo.Var(
                    within=pyo.NonNegativeReals, bounds=(0, max_weight_bound)
                ),
            )

            def constraint_bound_w0(model, i, j):
                return getattr(model, w_l)[i, j] - self.w_orig[
                    i, j
                ] <= getattr(model, dw_l)

            def constraint_bound_w1(model, i, j):
                return getattr(model, w_l)[i, j] - self.w_orig[
                    i, j
                ] >= -getattr(model, dw_l)

            def constraint_bound_b0(model, j):
                return getattr(model, b_l)[j] - self.b_orig[j] <= getattr(
                    model, dw_l
                )

            def constraint_bound_b1(model, j):
                return getattr(model, b_l)[j] - self.b_orig[j] >= -getattr(
                    model, dw_l
                )

            setattr(
                self.model,
                "w_bounded_constraint0" + str(l),
                pyo.Constraint(
                    range(self.uin), range(self.uout), rule=constraint_bound_w0
                ),
            )
            setattr(
                self.model,
                "w_bounded_constraint1" + str(l),
                pyo.Constraint(
                    range(self.uin), range(self.uout), rule=constraint_bound_w1
                ),
            )
            setattr(
                self.model,
                "b_bounded_constraint0" + str(l),
                pyo.Constraint(range(self.uout), rule=constraint_bound_b0),
            )
            setattr(
                self.model,
                "b_bounded_constraint1" + str(l),
                pyo.Constraint(range(self.uout), rule=constraint_bound_b1),
            )
        ###############################################################

        return getattr(self.model, x_l)
