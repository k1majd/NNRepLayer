from typing import List, Union
import numpy.typing as npt
import numpy as np
from .dense import Dense


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

    def give_mlp_node_bounds(
        self,
        from_layer: int,
        input_data: npt.NDArray,
        weigh_purturb: Union[int, float],
    ):
        """Returns the bounds of the nodes from the from_layer to the end.
        Args:
            from_layer: Target Layer
            x_train: Training Data
        """
        self.give_nodes_active_status(from_layer, input_data, weigh_purturb)
        pass

    def give_nodes_active_status(
        self,
        from_layer: int,
        input_data: npt.NDArray,
        weigh_purturb: Union[int, float],
    ):
        """Returns the active status of the nodes from the from_layer to the end.
        Args:
            from_layer: Target Layer
            input_data: input Data. Shape = (n_samples, n_features)
        """
        layer_values = self.__call__(input_data)
        weights = self.get_mlp_weights()
        bias = self.get_mlp_biases()

        intervals = []
        # get the intervals of the inputs of from_layer layer
        interval_temp = []
        for node in range(self.architecture[from_layer - 1]):
            interval_temp.append(
                [
                    layer_values[from_layer - 1][:, node].min(),
                    layer_values[from_layer - 1][:, node].max(),
                ]
            )
        intervals.append(interval_temp)

        # get the intervals of the outputs of from_layer layer
        for layer in range(from_layer, len(self.architecture)):
            interval_temp = []
            for node_next in range(self.architecture[layer]):
                lb = bias[layer - 1][node_next] - weigh_purturb
                ub = bias[layer - 1][node_next] + weigh_purturb
                for node_prev in range(self.architecture[layer - 1]):
                    if layer == from_layer:
                        if weights[layer - 1][node_prev][node_next] >= 0:
                            w_temp = (
                                weights[layer - 1][node_prev][node_next]
                                + weigh_purturb
                            )
                        else:
                            w_temp = (
                                weights[layer - 1][node_prev][node_next]
                                - weigh_purturb
                            )
                    else:
                        w_temp = weights[layer - 1][node_prev][node_next]

                    lb += intervals[0][node_prev][0] * max(
                        w_temp, 0
                    ) + intervals[0][node_prev][1] * min(w_temp, 0)
                    ub += intervals[0][node_prev][1] * max(
                        w_temp, 0
                    ) + intervals[0][node_prev][0] * min(w_temp, 0)
                interval_temp.append([lb, ub])
            intervals.append(interval_temp)

        print("here")

        pass

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
