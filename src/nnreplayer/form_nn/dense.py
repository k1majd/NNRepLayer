# Added seed. check

from multiprocessing.sharedctypes import Value
import numpy as np
import numpy.typing as npt
from typing import Optional

class Dense:
    """_summary_"""

    def __init__(self, n_in:int, n_out:int, relu:bool=False, seed:int = 12345) -> None:
        """_summary_

        Args:
            nin (int): _description_
            nout (int): _description_
            relu (bool, optional): _description_. Defaults to False.
        """
        if n_in <= 0:
            raise ValueError(f"n_in must be > 0. Received {n_in} instead")
        if n_out <= 0:
            raise ValueError(f"n_out must be > 0. Received {n_out} instead")
        rng = np.random.default_rng(seed)
        self.weights = rng.random((n_in, n_out)) * 2 - 1
        self.bias = rng.random((n_out)) * 2 - 1
        self.relu = relu

    def _relu(self, x:npt.NDArray) -> npt.NDArray:
        """_summary_

        Args:
            x (npt.NDArray): _description_

        Returns:
            npt.NDArray: _description_
        """

        return np.maximum(x, 0)

    def __call__(self, x:npt.NDArray) -> npt.NDArray:
        """_summary_

        Args:
            x (npt.NDArray): _description_

        Returns:
            npt.NDArray: _description_
        """

        x = x @ self.weights + self.bias
        if self.relu:
            x = self._relu(x)

        return x

    def set_variables(self, weights:Optional[npt.NDArray]=None, bias:Optional[npt.NDArray]=None) -> None:
        """_summary_

        Args:
            weights (_type_, optional): _description_. Defaults to None.
            bias (_type_, optional): _description_. Defaults to None.
        """
        if weights is not None:
            self.weights[:] = weights[:]

        if bias is not None:
            self.bias[:] = bias[:]
