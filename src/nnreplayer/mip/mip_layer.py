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
        w_error_norm: int = 0,
        # weight_activations: npt.NDArray,
        # bias_activations: npt.NDArray,
        # max_weight_bound: Union[int, float],
        ####################################
        param_bounds: tuple = (-1, 1),
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

        self.layer_num_next = (
            getattr(self, "layer_num", 0) + 1
        )  # next layer number
        if relu:
            # define relu layer constraints
            x_next = self._relu_constraints(
                x, shape, max_weight_bound, output_bounds
            )
        else:
            # define linear layer constraints
            x_next = self._constraints(
                x,
                shape,
                # self.layer_num_next,
                output_constraint_list,
                max_weight_bound,
                output_bounds,
            )

        # define w bounding constraints
        if self.layer_num == self.layer_to_repair:
            print(f"Repairing {self.layer_to_repair}th layer")
            if self.w_error_norm == 0:  # l-inf norm error
                self._weight_bound_constraint_linf(max_weight_bound)
            else:  # l-1 norm error
                self._weight_bound_constraint_l1(max_weight_bound)
        return x_next

    def _relu_constraints(
        self,
        x: npt.NDArray,
        shape: tuple,
        # l,
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

        x_l, s_l, theta_l = (
            "x" + str(self.layer_num_next),
            "s" + str(self.layer_num_next),
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
            "eq_constraint" + str(self.layer_num_next),
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
            "disjunction" + str(self.layer_num_next),
            pyg.Disjunction(
                range(m), range(num_next_repair_nodes), rule=disjuncts
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

    def _constraints(
        self,
        x: npt.NDArray,
        shape: tuple,
        # l,
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

        x_l = "x" + str(self.layer_num_next)

        # x_l = 'x'+str(self.layer_num_next)
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
            "eq_constraint" + str(self.layer_num_next),
            pyo.Constraint(range(m), range(self.uout), rule=constraints),
        )

        if output_constraint_list:
            constraint_addition_string = generate_output_constraints(
                output_constraint_list
            )
            exec(constraint_addition_string, locals(), globals())

        return getattr(self.model, x_l)

    def _weight_bound_constraint_l1(self, max_weight_bound):
        w_l = "w" + str(self.layer_num)
        b_l = "b" + str(self.layer_num)

        dw_l = "dw"
        setattr(
            self.model,
            dw_l,
            pyo.Var(within=pyo.NonNegativeReals, bounds=(0, max_weight_bound)),
        )
        # add the L1 bounding constraints for the nodes
        dw_pos_l = "dw_pos"
        dw_neg_l = "dw_neg"
        dw_int_l = "dw_int"

        setattr(
            self.model,
            dw_pos_l,
            pyo.Var(
                range(self.uin),
                range(len(self.repair_node_list)),
                domain=pyo.NonNegativeReals,
                bounds=(0, max_weight_bound),
            ),
        )
        setattr(
            self.model,
            dw_neg_l,
            pyo.Var(
                range(self.uin),
                range(len(self.repair_node_list)),
                domain=pyo.NonNegativeReals,
                bounds=(0, max_weight_bound),
            ),
        )

        setattr(
            self.model,
            dw_int_l,
            pyo.Var(
                range(self.uin),
                range(len(self.repair_node_list)),
                domain=pyo.Binary,
            ),
        )

        def constraint_bound_equiality_w(model, i, j):
            return (
                getattr(model, w_l)[i, self.repair_node_list.index(j)]
                - self.w_orig[i, j]
                == getattr(model, dw_pos_l)[i, self.repair_node_list.index(j)]
                - getattr(model, dw_neg_l)[i, self.repair_node_list.index(j)]
            )

        setattr(
            self.model,
            "w_equality_bound_l1_constraint_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(self.uin),
                self.repair_node_list,
                rule=constraint_bound_equiality_w,
            ),
        )

        def constraint_positive_constraint_w(model, i, j):
            return (
                getattr(model, dw_pos_l)[i, self.repair_node_list.index(j)]
                <= max_weight_bound
                * getattr(model, dw_int_l)[i, self.repair_node_list.index(j)]
            )

        setattr(
            self.model,
            "w_positive_bound_l1_constraint_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(self.uin),
                self.repair_node_list,
                rule=constraint_positive_constraint_w,
            ),
        )

        def constraint_negative_constraint_w(model, i, j):
            return getattr(model, dw_neg_l)[
                i, self.repair_node_list.index(j)
            ] <= max_weight_bound * (
                1 - getattr(model, dw_int_l)[i, self.repair_node_list.index(j)]
            )

        setattr(
            self.model,
            "w_negative_bound_l1_constraint_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(self.uin),
                self.repair_node_list,
                rule=constraint_negative_constraint_w,
            ),
        )

        db_pos_l = "db_pos"
        db_neg_l = "db_neg"
        db_int_l = "db_int"

        setattr(
            self.model,
            db_pos_l,
            pyo.Var(
                range(len(self.repair_node_list)),
                domain=pyo.NonNegativeReals,
                bounds=(0, max_weight_bound),
            ),
        )
        setattr(
            self.model,
            db_neg_l,
            pyo.Var(
                range(len(self.repair_node_list)),
                domain=pyo.NonNegativeReals,
                bounds=(0, max_weight_bound),
            ),
        )

        setattr(
            self.model,
            db_int_l,
            pyo.Var(
                range(len(self.repair_node_list)),
                domain=pyo.Binary,
            ),
        )

        def constraint_bound_equiality_b(model, j):
            return (
                getattr(model, b_l)[self.repair_node_list.index(j)]
                - self.b_orig[j]
                == getattr(model, db_pos_l)[self.repair_node_list.index(j)]
                - getattr(model, db_neg_l)[self.repair_node_list.index(j)]
            )

        setattr(
            self.model,
            "b_equality_bound_l1_constraint_lay" + str(self.layer_num_next),
            pyo.Constraint(
                self.repair_node_list,
                rule=constraint_bound_equiality_b,
            ),
        )

        def constraint_positive_constraint_b(model, j):
            return (
                getattr(model, db_pos_l)[self.repair_node_list.index(j)]
                <= max_weight_bound
                * getattr(model, db_int_l)[self.repair_node_list.index(j)]
            )

        setattr(
            self.model,
            "b_positive_bound_l1_constraint_lay" + str(self.layer_num_next),
            pyo.Constraint(
                self.repair_node_list,
                rule=constraint_positive_constraint_b,
            ),
        )

        def constraint_negative_constraint_b(model, j):
            return getattr(model, db_neg_l)[
                self.repair_node_list.index(j)
            ] <= max_weight_bound * (
                1 - getattr(model, db_int_l)[self.repair_node_list.index(j)]
            )

        setattr(
            self.model,
            "b_negative_bound_l1_constraint_lay" + str(self.layer_num_next),
            pyo.Constraint(
                self.repair_node_list,
                rule=constraint_negative_constraint_b,
            ),
        )

        # summation constraint

        def constraint_sum_positive_negative(model):
            return sum(
                getattr(model, dw_pos_l)[i, j]
                for i in range(self.uin)
                for j in range(len(self.repair_node_list))
            ) + sum(
                getattr(model, dw_neg_l)[i, j]
                for i in range(self.uin)
                for j in range(len(self.repair_node_list))
            ) + sum(
                getattr(model, db_pos_l)[j]
                for j in range(len(self.repair_node_list))
            ) + sum(
                getattr(model, db_neg_l)[j]
                for j in range(len(self.repair_node_list))
            ) <= getattr(
                model, dw_l
            )

        setattr(
            self.model,
            "w_b_sum_bound_l1_constraint_lay" + str(self.layer_num_next),
            pyo.Constraint(rule=constraint_sum_positive_negative),
        )

    def _weight_bound_constraint_linf(self, max_weight_bound):

        w_l = "w" + str(self.layer_num)
        b_l = "b" + str(self.layer_num)

        dw_l = "dw"
        setattr(
            self.model,
            dw_l,
            pyo.Var(within=pyo.NonNegativeReals, bounds=(0, max_weight_bound)),
        )

        # add the L-inf bounding constraints for the weights
        def constraint_bound_w_upper(model, i, j):
            return getattr(model, w_l)[
                i, self.repair_node_list.index(j)
            ] - self.w_orig[i, j] <= getattr(model, dw_l)

        def constraint_bound_w_lower(model, i, j):
            return getattr(model, w_l)[
                i, self.repair_node_list.index(j)
            ] - self.w_orig[i, j] >= -getattr(model, dw_l)

        def constraint_bound_b_upper(model, j):
            return getattr(model, b_l)[
                self.repair_node_list.index(j)
            ] - self.b_orig[j] <= getattr(model, dw_l)

        def constraint_bound_b_lower(model, j):
            return getattr(model, b_l)[
                self.repair_node_list.index(j)
            ] - self.b_orig[j] >= -getattr(model, dw_l)

        setattr(
            self.model,
            "w_lower_bound_linf_constrain_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(self.uin),
                self.repair_node_list,
                rule=constraint_bound_w_upper,
            ),
        )
        setattr(
            self.model,
            "w_upper_bound_linf_constrain_lay" + str(self.layer_num_next),
            pyo.Constraint(
                range(self.uin),
                self.repair_node_list,
                rule=constraint_bound_w_lower,
            ),
        )
        setattr(
            self.model,
            "b_upper_bound_linf_constrain_lay" + str(self.layer_num_next),
            pyo.Constraint(
                self.repair_node_list, rule=constraint_bound_b_upper
            ),
        )
        setattr(
            self.model,
            "b_lower_bound_linf_constrain_lay" + str(self.layer_num_next),
            pyo.Constraint(
                self.repair_node_list, rule=constraint_bound_b_lower
            ),
        )
