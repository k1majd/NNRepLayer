import numpy as np


class Dense:
    """_summary_"""

    def __init__(self, nin, nout):
        """_summary_

        Args:
            nin (_type_): _description_
            nout (_type_): _description_
        """
        self.weights = np.random.rand(nin, nout) * 2 - 1
        self.bias = np.random.rand(nout) * 2 - 1

    def _relu(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return np.maximum(x, 0)

    def __call__(self, x, relu=False):
        """_summary_

        Args:
            x (_type_): _description_
            relu (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        x = x @ self.weights + self.bias
        if relu:
            x = self._relu(x)

        return x

    def set_variables(self, weights=None, bias=None):
        """_summary_

        Args:
            weights (_type_, optional): _description_. Defaults to None.
            bias (_type_, optional): _description_. Defaults to None.
        """
        if weights is not None:
            self.weights[:] = weights[:]

        if bias is not None:
            self.bias[:] = bias[:]
