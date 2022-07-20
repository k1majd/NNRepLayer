import pyomo.environ as pyo
from pyomo.gdp import *
from .lp_layer import LPLayer
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from typing import Union


class LPNNModel:
    """_summary_"""

    def __init__(
        self,
        layer_to_repair: npt.NDArray,
        architecture: List[int],
        weights: List[npt.NDArray],
        bias: List[npt.NDArray],
        ####################################
        # TODO: add these parameters
        repair_node_list: List[int] = None,
        max_weight_bound: float = 2.0,
        param_precision: int = 6,
        data_precision: int = 6,
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

        self.model = pyo.AbstractModel()

        self.model.nlayers = layer_to_repair
        self.layer_to_repair = layer_to_repair

        self.uin, self.uout = (
            architecture[layer_to_repair - 1],
            architecture[-2],
        )
        uhidden = architecture[layer_to_repair:-1]

        self.layers = []
        prev = architecture[layer_to_repair - 1]
        ####################################
        # TODO: edit this part for inputting the target repair nodes
        if repair_node_list is None:
            repair_node_list = list(range(architecture[layer_to_repair]))

        num_layers_ahead = len(architecture) - self.model.nlayers - 2
        # print("UHidden = {}".format(uhidden))
        for iterate, u in enumerate(uhidden):
            self.layers.append(
                LPLayer(
                    self.model,
                    layer_to_repair,
                    prev,
                    u,
                    weights[layer_to_repair - 1 + iterate],
                    bias[layer_to_repair - 1 + iterate],
                    # TODO: add these parameters
                    num_layers_ahead,
                    repair_node_list,
                    max_weight_bound,
                    param_precision,
                )
            )
            num_layers_ahead = len(architecture) - self.model.nlayers - 1
            prev = u

    def __call__(
        self,
        final_layer,
        final_node,
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
        layer = self.layers[0]
        x = layer(final_layer, final_node)
        for layers in self.layers[1:]:
            x = layers(final_layer, final_node, x)

        return x
        # for layer in self.layers[:-1]:
        #     ####################################
        #     # TODO: (12_7_2022) detect if upper and lower bounds are given
        #     if nodes_upper is not None:
        #         ub = nodes_upper[bound_idx]
        #         lb = nodes_lower[bound_idx]
        #     else:
        #         ub = np.ones((m, n)) * output_bounds[1]
        #         lb = np.ones((m, n)) * output_bounds[0]
        #     bound_idx += 1
        #     ######################################
        #     x = layer(
        #         x,
        #         (m, layer.uin),
        #         output_constraint_list,
        #         ##############################
        #         # TODO: (12_7_2022) input the upper and lower bounds of nodes for
        #         # each sample
        #         ub,
        #         lb,
        #         ##############################
        #         relu=True,
        #         max_weight_bound=max_weight_bound,
        #         output_bounds=output_bounds,
        #     )

        # layer = self.layers[-1]
        # ####################################
        # # TODO: (12_7_2022) detect if upper and lower bounds are given
        # if nodes_upper is not None:
        #     ub = nodes_upper[bound_idx]
        #     lb = nodes_lower[bound_idx]
        # else:
        #     ub = np.ones((m, n)) * output_bounds[1]
        #     lb = np.ones((m, n)) * output_bounds[0]
        # ######################################
        # y = layer(
        #     x,
        #     (m, layer.uin),
        #     output_constraint_list,
        #     ##############################
        #     # TODO: (12_7_2022) input the upper and lower bounds of nodes for
        #     # each sample
        #     ub,
        #     lb,
        #     ##############################
        #     relu=relu,
        #     max_weight_bound=max_weight_bound,
        #     output_bounds=output_bounds,
        # )
        # return y
