from typing import List, Union
import numpy.typing as npt
import numpy as np
from .dense import Dense
from nnreplayer.utils import BoundStatTracker


class MLP:
    """Defines Structure of Input Network intended to Repair."""

    def __init__(
        self, n_in: int, n_out: int, n_hidden: List, relu=False
    ) -> None:
        """Initializes the structure of the Neural Network

        Args:
            n_in: Input Layer Size
            n_out: Output Layer Size
            n_hidden: Hidden Layer Size
            relu: If true, Applies ReLU Activation Function. Defaults to False.
        """
        self.architecture = [n_in] + n_hidden + [n_out]
        self.num_layer = len(n_hidden) + 1
        prev = n_in
        self.layers = []
        for hidden in n_hidden:
            self.layers.append(Dense(prev, hidden, relu=True))
            prev = hidden
        self.layers.append(Dense(prev, n_out, relu=relu))

        # TODO: bound stat tracker properties
        self.bound_stat_tracker = BoundStatTracker(self.architecture)

    def __call__(self, input_data: npt.NDArray) -> npt.NDArray:
        """Perform Feed Forward pass using Input

        Args:
            input_data: Input Data

        Returns:
            Output Data
        """
        layer_values = [input_data]
        for layer in self.layers:
            input_data = layer(input_data)
            layer_values.append(input_data)
        return layer_values

    # TODO: give bound nodes method
    def give_nodes_bounds(
        self,
        layer_to_repair: int,
        input_data: npt.NDArray,
        max_weight_bound: Union[int, float],
        repair_node_list: List[int],
    ):
        """Returns the active status of the nodes from the layer_to_repair to the end.
        Args:
            layer_to_repair: Target Layer
            input_data: input Data. Shape = (n_samples, n_features)
        """
        layer_values = self.__call__(input_data)
        weights = self.get_mlp_weights()
        bias = self.get_mlp_biases()

        ub_mat = []
        lb_mat = []
        # get the intervals for the layer_to_repair layer outputs
        ub = np.zeros(
            (input_data.shape[0], self.architecture[layer_to_repair])
        )
        lb = np.zeros(
            (input_data.shape[0], self.architecture[layer_to_repair])
        )
        for node_next in range(self.architecture[layer_to_repair]):
            for s in range(input_data.shape[0]):
                if node_next in repair_node_list:
                    lb[s, node_next] = (
                        bias[layer_to_repair - 1][node_next] - max_weight_bound
                    )
                    ub[s, node_next] = (
                        bias[layer_to_repair - 1][node_next] + max_weight_bound
                    )
                    for node_prev in range(
                        self.architecture[layer_to_repair - 1]
                    ):
                        lb[s, node_next] += (
                            weights[layer_to_repair - 1][node_prev][node_next]
                            - max_weight_bound
                        ) * max(
                            0.0,
                            layer_values[layer_to_repair - 1][s][node_prev],
                        ) + (
                            weights[layer_to_repair - 1][node_prev][node_next]
                            + max_weight_bound
                        ) * min(
                            0.0,
                            layer_values[layer_to_repair - 1][s][node_prev],
                        )

                        ub[s, node_next] += (
                            weights[layer_to_repair - 1][node_prev][node_next]
                            + max_weight_bound
                        ) * max(
                            0.0,
                            layer_values[layer_to_repair - 1][s][node_prev],
                        ) + (
                            weights[layer_to_repair - 1][node_prev][node_next]
                            - max_weight_bound
                        ) * min(
                            0.0,
                            layer_values[layer_to_repair - 1][s][node_prev],
                        )
                    # update stats
                    self.bound_stat_tracker.update_stats(
                        lb[s, node_next],
                        ub[s, node_next],
                        layer_to_repair,
                        node_next,
                    )
                else:
                    lb[s, node_next] = layer_values[layer_to_repair][s][
                        node_next
                    ]
                    ub[s, node_next] = layer_values[layer_to_repair][s][
                        node_next
                    ]
        # print stats
        self.bound_stat_tracker.print_stats(layer_to_repair)

        ub_mat.append(ub)
        lb_mat.append(lb)

        # get the intervals for the subsequent layers
        for layer in range(layer_to_repair + 1, len(self.architecture)):
            ub = np.zeros((input_data.shape[0], self.architecture[layer]))
            lb = np.zeros((input_data.shape[0], self.architecture[layer]))
            for node_next in range(self.architecture[layer]):
                for s in range(input_data.shape[0]):
                    lb[s, node_next] = (
                        bias[layer - 1][node_next] - max_weight_bound
                    )
                    ub[s, node_next] = (
                        bias[layer - 1][node_next] + max_weight_bound
                    )
                    for node_prev in range(self.architecture[layer - 1]):
                        w_temp = weights[layer - 1][node_prev][node_next]
                        lb[s, node_next] += lb_mat[
                            layer - layer_to_repair - 1
                        ][s][node_prev] * max(w_temp, 0) + ub_mat[
                            layer - layer_to_repair - 1
                        ][
                            s
                        ][
                            node_prev
                        ] * min(
                            w_temp, 0
                        )
                        ub[s, node_next] += ub_mat[
                            layer - layer_to_repair - 1
                        ][s][node_prev] * max(w_temp, 0) + lb_mat[
                            layer - layer_to_repair - 1
                        ][
                            s
                        ][
                            node_prev
                        ] * min(
                            w_temp, 0
                        )
                    # update stats
                    self.bound_stat_tracker.update_stats(
                        lb[s, node_next],
                        ub[s, node_next],
                        layer,
                        node_next,
                    )
            # print stats
            self.bound_stat_tracker.print_stats(layer)
            ub_mat.append(ub)
            lb_mat.append(lb)

        return ub_mat, lb_mat

    def set_mlp_params(self, mlp_weights: List[npt.NDArray]) -> None:
        """Manually set Weights And Bias Parameters for entire network

        Args:
            mlp_weights: List of [weight(layer), bias(layer)] for layer = 1,2,3,...
        """

        iterate = 0
        num_lays = len(self.layers)
        for j in range(num_lays):
            self.layers[j].weights = mlp_weights[iterate]
            iterate += 1
            self.layers[j].bias = mlp_weights[iterate]
            iterate += 1

    def set_mlp_params_layer(self, mlp_weights: List[npt.NDArray], layer: int):
        """Manually set Weights And Bias Parameters of a Particular Layer

        Args:
            mlp_weights: List of [weight(layer), bias(layer)] for layer = 1,2,3,...
            layer: Target Layer for layer = 1,2,3,...

        Raises:
            ValueError: If Target Layer outside the architecture of the neural network.
        """
        if not (layer <= len(self.layers) and layer >= 1):
            raise ValueError(
                f"Layer to repair out of bounds. Expected [{1}, {len(self.layers)}]. Received {layer} instead."
            )

        self.layers[layer - 1].weights = mlp_weights[0]
        self.layers[layer - 1].bias = mlp_weights[1]

    def get_mlp_params(self) -> None:
        """Returns all Weights and Bias Parameters of NN"""

        weight_bias_list = []
        for layer in self.layers:
            weight_bias_list.append(layer.weights)
            weight_bias_list.append(layer.bias)

        return weight_bias_list

    def get_mlp_weights(self) -> List[npt.NDArray]:
        """Returns all Weights Parameters of NN

        Returns:
            List of [weight(layer)] for layer = 1,2,3,...
        """

        weight_list = []
        for layer in self.layers:
            weight_list.append(layer.weights)

        return weight_list

    def get_mlp_biases(self) -> List[npt.NDArray]:
        """Returns all Bias Parameters of NN

        Returns:
            List of [bias(layer)] for layer = 1,2,3,...
        """

        bias_list = []
        for layer in self.layers:
            bias_list.append(layer.bias)

        return bias_list

    def get_mlp_params_layer(self, layer: int) -> npt.NDArray:
        """Returns Weights and Bias Parameters for a Particular Layer

        Args:
            Target Layer

        Raises:
            ValueError: If Target Layer outside the architecture of the neural network.

        Returns:
            List of [weights(layer), bias(layer)] for layer = 1,2,3,...
        """
        if not (layer <= len(self.layers) and layer >= 1):
            raise ValueError(
                f"Layer to repair out of bounds. Expected [{1}, {len(self.layers)}]. Received {layer} instead."
            )

        return [self.layers[layer - 1].weights, self.layers[layer - 1].bias]

    def get_mlp_weight_layer(self, layer: int) -> npt.NDArray:
        """Returns Weights Parameters of a Particular Layer

        Args:
            layer: Target Layer

        Raises:
            ValueError: If Target Layer outside the architecture of the neural network.

        Returns:
            List of [weights(layer)] for layer = 1,2,3,...
        """
        if not (layer <= len(self.layers) and layer >= 1):
            raise ValueError(
                f"Layer to repair out of bounds. Expected [{1}, {len(self.layers)}]. Received {layer} instead."
            )

        return self.layers[layer - 1].weights

    def get_mlp_bias_layer(self, layer: int) -> npt.NDArray:
        """Returns Bias Parameters of a Particular Layer

        Args:
            layer: Target Layer

        Raises:
            ValueError: If Target Layer outside the architecture of the neural network.

        Returns:
            List of [bias(layer)] for layer = 1,2,3,...
        """
        if not (layer <= len(self.layers) and layer >= 1):
            raise ValueError(
                f"Layer to repair out of bounds. Expected [{1}, {len(self.layers)}]. Received {layer} instead."
            )

        return self.layers[layer - 1].bias
