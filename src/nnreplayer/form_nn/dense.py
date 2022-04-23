from typing import Optional
import numpy as np
import numpy.typing as npt


class Dense:
    """Defines a dense feed forward layer.
    """

    def __init__(
        self, n_in: int, n_out: int, relu: bool = False, seed: int = 12345
    ) -> None:
        """Intializes a Dense layer with random weights and relu activation if enabled

        Args:
            nin:  Number of input Nodes
            nout: Number of output Nodes
            relu: Relu Activation Applied to layer or not Defaults to False.
            seed: Seed to define Random weights. Defaults to 12345.

        Raises:
            ValueError: n_in should be > 0
            ValueError: n_out should be > 0
        """

        if n_in <= 0:
            raise ValueError(f"n_in must be > 0. Received {n_in} instead")
        if n_out <= 0:
            raise ValueError(f"n_out must be > 0. Received {n_out} instead")
        rng = np.random.default_rng(seed)
        self.weights = rng.random((n_in, n_out)) * 2 - 1
        self.bias = rng.random((n_out)) * 2 - 1
        self.relu = relu

    def _relu(self, input_data: npt.NDArray) -> npt.NDArray:
        """Applies ReLU Activation Function

        Args:
            x: Input Data

        Returns:
            Output after applying ReLU Activation Function
        """

        return np.maximum(input_data, 0)

    def __call__(self, input_data: npt.NDArray) -> npt.NDArray:
        """Forward Pass through the Dense Layer.

        Args:
            x: Input Data

        Returns:
            Performs Linear Transform And Applies Activation if enabled
        """

        input_data = input_data @ self.weights + self.bias
        if self.relu:
            input_data = self._relu(input_data)

        return input_data

    def set_variables(
        self,
        weights: Optional[npt.NDArray] = None,
        bias: Optional[npt.NDArray] = None,
    ) -> None:
        """Helper Function to manually set the Weights and Bias of the Dense Layer

        Args:
            weights: Weights of a layer. Defaults to None.
            bias: Bias of a Neural Network. Defaults to None.
        """
        if weights is not None:
            self.weights[:] = weights[:]

        if bias is not None:
            self.bias[:] = bias[:]
