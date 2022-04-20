import pyomo.environ as pyo
from pyomo.gdp import *
from .mip_layer import MIPLayer
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from typing import Union


class MIPNNModel:
    """_summary_"""

    def __init__(
        self,
        layer_to_repair: npt.NDArray,
        architecture: List[int],
        weights: List[npt.NDArray],
        bias: List[npt.NDArray],
        param_bounds: tuple = (-1, 1),
    ):
        """_summary_

        Args:
            layer_to_repair (npt.NDArray): _description_
            architecture (List[int]): _description_
            weights (List[npt.NDArray]): _description_
            bias (List[npt.NDArray]): _description_
            param_bounds (tuple, optional): _description_. Defaults to (-1, 1).
        """

        self.model = pyo.ConcreteModel()

        self.model.nlayers = layer_to_repair

        self.uin, self.uout = (
            architecture[layer_to_repair - 1],
            architecture[-1],
        )
        uhidden = architecture[layer_to_repair:-1]

        self.layers = []
        prev = architecture[layer_to_repair - 1]
        # print("UHidden = {}".format(uhidden))
        for iterate, u in enumerate(uhidden):
            self.layers.append(
                MIPLayer(
                    self.model,
                    layer_to_repair,
                    prev,
                    u,
                    weights[layer_to_repair - 1 + iterate],
                    bias[layer_to_repair - 1 + iterate],
                    param_bounds,
                )
            )
            prev = u
        self.layers.append(
            MIPLayer(
                self.model,
                layer_to_repair,
                prev,
                architecture[-1],
                weights[-1],
                bias[-1],
                param_bounds,
            )
        )

    def __call__(
        self,
        x: npt.NDArray,
        shape: Tuple,
        output_constraint_list: List[npt.NDArray],
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

        m, n = shape
        assert n == self.uin

        for layer in self.layers[:-1]:
            x = layer(
                x,
                (m, layer.uin),
                output_constraint_list,
                relu=True,
                max_weight_bound=max_weight_bound,
                output_bounds=output_bounds,
            )

        layer = self.layers[-1]
        y = layer(
            x,
            (m, layer.uin),
            output_constraint_list,
            relu=relu,
            max_weight_bound=max_weight_bound,
            output_bounds=output_bounds,
        )
        return y
