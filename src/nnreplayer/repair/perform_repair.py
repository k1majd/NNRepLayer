from ..utils.results import Results
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow import keras
import tensorflow.keras.backend as kb

from .repair_weights_class import repair_weights


class NNRepLayer(repair_weights):
    """_summary_

    Args:
        repair_weights (_type_): _description_
    """

    def __init__(
        self,
        model_orig,
        layer_to_repair,
        architecture,
        output_constraint_list,
        squared_sum,
    ):
        """_summary_

        Args:
            model_orig (_type_): _description_
            layer_to_repair (_type_): _description_
            architecture (_type_): _description_
            output_constraint_list (_type_): _description_
            squared_sum (_type_): _description_
        """
        self.opt_model = None
        super().__init__(
            model_orig,
            layer_to_repair,
            architecture,
            output_constraint_list,
            squared_sum,
        )

    def display_repair_opt_model(self, train_dataset, options):
        """_summary_

        Args:
            train_dataset (_type_): _description_
            options (_type_): _description_
        """
        x_train, y_train = train_dataset
        layer_values_train = super().extract_network(x_train)

        _, opt_model = super().set_up_optimizer(
            y_train, layer_values_train, max_weight_bound=options.max_weight_bound
        )

    def display_opt_solved_model(self):
        """_summary_

        Raises:
            AttributeError: _description_
        """
        if hasattr(self, "opt_model_solved"):
            self.opt_model_solved.display()
        else:
            raise AttributeError(
                "'NNRepLayer' object has no attribute 'opt_model_solved': run 'perform_repair' method first!"
            )

    def generate_repair_opt_model(self, train_dataset, options, format="lp"):
        """TO DO

        Args:
            train_dataset (_type_): _description_
            options (_type_): _description_
            format (str, optional): _description_. Defaults to "lp".
        """
        pass

    def give_opt_model(self, train_dataset, options):
        """_summary_

        Args:
            train_dataset (_type_): _description_
            options (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_train, y_train = train_dataset
        layer_values_train = super().extract_network(x_train)

        _, opt_model = super().set_up_optimizer(
            y_train, layer_values_train, max_weight_bound=options.max_weight_bound
        )
        return opt_model

    def perform_repair(self, train_dataset, options):
        """_summary_

        Args:
            train_dataset (_type_): _description_
            options (_type_): _description_
        """
        x_train, y_train = train_dataset
        layer_values_train = super().extract_network(x_train)

        _, opt_model = super().set_up_optimizer(
            y_train, layer_values_train, max_weight_bound=options.max_weight_bound
        )
        opt_model_solved = super().solve_optimization_problem(
            opt_model,
            # cost_expr,
            options.gdp_formulation,
            options.solver_factory,
            options.solver_language,
            options.optimizer_options,
        )
        self.opt_model_solved = opt_model_solved

        return self.give_results(train_dataset, opt_model_solved, options)

    def give_results(self, train_dataset, opt_model_solved, options):
        """_summary_

        Args:
            train_dataset (_type_): _description_
            opt_model_solved (_type_): _description_
            options (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_train, y_train = train_dataset
        model_new_params = super().set_new_params(opt_model_solved)

        new_model = super().return_repaired_model(
            model_new_params, options.model_output_type
        )
        weights = [
            self.model_orig_params[iterate]
            for iterate in range(0, 2 * (len(self.architecture) - 1), 2)
        ]
        bias = [
            self.model_orig_params[iterate]
            for iterate in range(1, 2 * (len(self.architecture) - 1), 2)
        ]

        new_weight = [
            model_new_params[iterate]
            for iterate in range(0, 2 * (len(self.architecture) - 1), 2)
        ]
        new_bias = [
            model_new_params[iterate]
            for iterate in range(1, 2 * (len(self.architecture) - 1), 2)
        ]

        y_new_train = new_model.predict(x_train)

        y_train_original = self.model_orig.predict(x_train)

        # mse_original_nn_train = self.give_mse_error(y_train, y_train_original)
        # mse_new_nn_train = self.give_mse_error(y_train, y_new_train)

        weight_error = np.max(
            new_weight[self.layer_to_repair - 1] - weights[self.layer_to_repair - 1]
        )
        bias_error = np.max(
            new_bias[self.layer_to_repair - 1] - bias[self.layer_to_repair - 1]
        )

        return Results(
            weights,
            bias,
            new_weight,
            new_bias,
            new_model,
            self.give_mse_error(y_train, y_train_original),
            self.give_mse_error(y_train, y_new_train),
            weight_error,
            bias_error,
        )

    def give_mse_error(self, data1, data2):
        """_summary_

        Args:
            data1 (_type_): _description_
            data2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        row, col = np.array(data1).shape
        _squared_sum = 0
        for i in range(row):
            for j in range(col):
                _squared_sum += (data1[i, j] - data2[i, j]) ** 2

        return _squared_sum / row


def perform_repair(
    layer_to_repair,
    model_orig,
    architecture,
    output_constraint_list,
    cost_function,
    train_dataset,
    options,
):
    """_summary_

    Args:
        layer_to_repair (_type_): _description_
        model_orig (_type_): _description_
        architecture (_type_): _description_
        output_constraint_list (_type_): _description_
        cost_function (_type_): _description_
        train_dataset (_type_): _description_
        options (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_train, y_train = train_dataset
    gdp_formulation = options.gdp_formulation
    solver_factory = options.solver_factory
    solver_language = options.solver_language
    model_output_type = options.model_output_type
    optimizer_options = options.optimizer_options
    opt_log_path = options.optimizer_log_path
    # optimizer_time_limit = options.optimizer_time_limit
    # optimizer_mip_gap = options.optimizer_mip_gap

    rep_weights = repair_weights(
        model_orig, layer_to_repair, architecture, output_constraint_list, cost_function
    )
    layer_values_train = rep_weights.extract_network(x_train)

    cost_expr, model_lay = rep_weights.set_up_optimizer(
        y_train, layer_values_train, max_weight_bound=options.max_weight_bound
    )

    new_model_lay = rep_weights.solve_optimization_problem(
        model_lay,
        cost_expr,
        gdp_formulation,
        solver_factory,
        solver_language,
        optimizer_options,
        opt_log_path,
    )
    model_new_params = rep_weights.set_new_params(new_model_lay)

    new_model = rep_weights.return_repaired_model(model_new_params, model_output_type)

    weights = [
        rep_weights.model_orig_params[iterate]
        for iterate in range(0, 2 * (len(architecture) - 1), 2)
    ]
    bias = [
        rep_weights.model_orig_params[iterate]
        for iterate in range(1, 2 * (len(architecture) - 1), 2)
    ]

    new_weight = [
        model_new_params[iterate]
        for iterate in range(0, 2 * (len(architecture) - 1), 2)
    ]
    new_bias = [
        model_new_params[iterate]
        for iterate in range(1, 2 * (len(architecture) - 1), 2)
    ]

    y_new_train = new_model.predict(x_train)

    y_train_original = model_orig.predict(x_train)

    MSE_original_nn_train = cost_function(y_train, y_train_original) / y_train.shape[0]
    MSE_new_nn_train = cost_function(y_train, y_new_train) / y_train.shape[0]

    weight_error = np.max(
        new_weight[layer_to_repair - 1] - weights[layer_to_repair - 1]
    )
    bias_error = np.max(new_bias[layer_to_repair - 1] - bias[layer_to_repair - 1])

    results = Results(
        weights,
        bias,
        new_weight,
        new_bias,
        new_model,
        MSE_original_nn_train,
        MSE_new_nn_train,
        weight_error,
        bias_error,
    )
    return results
