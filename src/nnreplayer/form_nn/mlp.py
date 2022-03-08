from .dense import Dense


class MLP:
    """_summary_"""

    def __init__(self, nin, uout, uhidden):
        """_summary_

        Args:
            nin (_type_): _description_
            uout (_type_): _description_
            uhidden (_type_): _description_
        """
        self.num_layer = len(uhidden) + 1
        prev = nin
        self.layers = []
        for u in uhidden:
            self.layers.append(Dense(prev, u))
            prev = u
        self.layers.append(Dense(prev, uout))

    def __call__(self, x, relu=False):
        """_summary_

        Args:
            x (_type_): _description_
            relu (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        x1 = self.layers[0](x, relu=True)
        for iterate in range(1, self.num_layer - 1):
            variable1 = "x" + str(iterate + 1)
            variable2 = "x" + str(iterate)
            vars()[variable1] = self.layers[iterate](vars()[variable2], relu=True)

        variable_name = "x" + str(self.num_layer - 1)
        y = self.layers[self.num_layer - 1](vars()[variable_name], relu=True)

        exec_string = ""
        for j in range(1, self.num_layer):
            exec_string = exec_string + "x" + str(j) + ", "
        exec_string = exec_string + "y"

        return list(eval(exec_string))

    def set_mlp_params(self, mlp_weights):
        """_summary_

        Args:
            mlp_weights (list of ndarray): a list of [wight(layer), bias(layer)] for layer = 1,2,3,...
        """

        iterate = 0
        num_lays = len(self.layers)
        for j in range(num_lays):
            self.layers[j].weights = mlp_weights[iterate]
            iterate = iterate + 1
            self.layers[j].bias = mlp_weights[iterate]
            iterate = iterate + 1

    def set_mlp_params_layer(self, mlp_weights, layer):
        """_summary_

        Args:
            mlp_weights (list of ndarray): a list of [weight, bias]
            layer (int): target layer i for i in 1,2,3,...
        """
        self.layers[layer - 1].weights = mlp_weights[0]
        self.layers[layer - 1].bias = mlp_weights[1]

    def get_mlp_params(self):
        """_summary_

        Returns:
            list of ndarray: a list of [wight(layer), bias(layer)] for layer = 1,2,3,...
        """

        weight_bias_list = []
        for layer in self.layers:
            weight_bias_list.append(layer.weights)
            weight_bias_list.append(layer.bias)

        return weight_bias_list

    def get_mlp_weights(self):
        """_summary_

        Returns:
            list of ndarray: a list of [wight(layer)] for layer = 1,2,3,...
        """

        weight_list = []
        for layer in self.layers:
            weight_list.append(layer.weights)

        return weight_list

    def get_mlp_biases(self):
        """_summary_

        Returns:
            list of ndarray: a list of [bias(layer)] for layer = 1,2,3,...
        """

        bias_list = []
        for layer in self.layers:
            bias_list.append(layer.bias)

        return bias_list

    def get_mlp_params_layer(self, layer):
        """_summary_

        Args:
            layer (int): target layer i for i in 1,2,3,...

        Returns:
            list of ndarray: a list of [weight(layer), bias(layer)]
        """

        return [self.layers[layer - 1].weights, self.layers[layer - 1].bias]

    def get_mlp_weight_layer(self, layer):
        """_summary_

        Args:
            layer (int): target layer i for i in 1,2,3,...

        Returns:
            ndarray: weight(layer)
        """

        return self.layers[layer - 1].weights

    def get_mlp_bias_layer(self, layer):
        """_summary_

        Args:
            layer (int): target layer i for i in 1,2,3,...

        Returns:
            ndarray: bias(layer)
        """

        return self.layers[layer - 1].bias
