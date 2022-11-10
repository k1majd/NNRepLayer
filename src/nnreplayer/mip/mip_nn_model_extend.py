import pyomo.environ as pyo
from pyomo.gdp import *
from .mip_layer_extend import MIPLayerExtend
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from typing import Union


class MIPNNModelExtend:
    """_summary_"""

    def __init__(
        self,
        architecture: List[int],
        weights: List[npt.NDArray],
        bias: List[npt.NDArray],
        opt_model: pyo.ConcreteModel,
        last_ctrl_layer: int,
        layer_num: int,
        param_precision: int = 6,
        repair_node_list: List[int] = None,
    ):
        """_summary_

        Args:
            layer_to_repair (npt.NDArray): _description_
            architecture (List[int]): _description_
            weights (List[npt.NDArray]): _description_
            bias (List[npt.NDArray]): _description_
            repair_node_list (List[int], optional): _description_. Defaults to [].
            w_error_norm (int, optional): weight error norm type 0 = L-inf, 1 = L-1. Defaults to 0.
            param_bounds (tuple, optional): _description_. Defaults to (-1, 1).
        """

        self.model = opt_model
        self.model.nlayers = layer_num

        self.uin, self.uout = (
            architecture[0],
            architecture[-1],
        )
        uhidden = architecture[1:-1]

        self.layers = []
        prev = architecture[0]
        if repair_node_list is None:
            repair_node_list = list(range(architecture[0]))

        for iterate, u in enumerate(uhidden):
            self.layers.append(
                MIPLayerExtend(
                    self.model,
                    last_ctrl_layer,
                    prev,
                    u,
                    weights[iterate],
                    bias[iterate],
                    repair_node_list,
                    param_precision,
                )
            )

            prev = u
        self.layers.append(
            MIPLayerExtend(
                self.model,
                last_ctrl_layer,
                prev,
                architecture[-1],
                weights[-1],
                bias[-1],
                repair_node_list,
                param_precision,
            )
        )
        ####################################

    def __call__(
        self,
        x: npt.NDArray,
        shape: Tuple,
        output_constraint_list: List[npt.NDArray],
        input_order: List[str],
        ctrl_name: str,
        ##############################
        # TODO: (12_7_2022) input the upper and lower bounds of nodes for
        # each sample
        nodes_upper: List[npt.NDArray],
        nodes_lower: List[npt.NDArray],
        ##############################
        relu: bool = False,
        output_bounds: tuple = (-1e1, 1e1),
    ):
        m, n = shape
        assert n == self.uin

        # construct input vector
        x = self.construct_input_vector(x, shape, input_order, ctrl_name)
        bound_idx = 0

        for layer in self.layers[:-1]:
            ####################################
            # TODO: (12_7_2022) detect if upper and lower bounds are given
            # if nodes_upper is not None:
            #     ub = nodes_upper[bound_idx]
            #     lb = nodes_lower[bound_idx]
            # else:
            #     ub = np.ones((m, n)) * output_bounds[1]
            #     lb = np.ones((m, n)) * output_bounds[0]
            ub = nodes_upper[bound_idx]
            lb = nodes_lower[bound_idx]
            bound_idx += 1
            ######################################
            x = layer(
                x,
                (m, layer.uin),
                output_constraint_list,
                ##############################
                # TODO: (12_7_2022) input the upper and lower bounds of nodes for
                # each sample
                ub,
                lb,
                ##############################
                relu=True,
                # output_bounds=output_bounds,
            )

        layer = self.layers[-1]
        ####################################
        # TODO: (12_7_2022) detect if upper and lower bounds are given
        # if nodes_upper is not None:
        ub = nodes_upper[bound_idx]
        lb = nodes_lower[bound_idx]
        # else:
        #     ub = np.ones((m, n)) * output_bounds[1]
        #     lb = np.ones((m, n)) * output_bounds[0]
        ######################################
        y = layer(
            x,
            (m, layer.uin),
            output_constraint_list,
            ##############################
            # TODO: (12_7_2022) input the upper and lower bounds of nodes for
            # each sample
            ub,
            lb,
            ##############################
            relu=relu,
            # output_bounds=output_bounds,
        )
        return y

    def construct_input_vector(self, x, shape, input_order, ctrl_name):
        m, n = shape
        assert n == self.uin

        x_next = []
        if input_order.index("state") == 0:
            for s in range(m):
                x_temp = []
                for c in range(x.shape[1]):
                    x_temp.append(x[s, c])
                for c in range(n - x.shape[1]):
                    x_temp.append(getattr(self.model, ctrl_name)[s, c])
                x_next.append(x_temp)
        else:
            for s in range(m):
                x_temp = []
                for c in range(n - x.shape[1]):
                    x_temp.append(getattr(self.model, ctrl_name)[s, c])
                for c in range(x.shape[1]):
                    x_temp.append(x[s, c])
                x_next.append(x_temp)

        return x_next
