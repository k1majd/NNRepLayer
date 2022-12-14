from dataclasses import dataclass
from platform import architecture
from typing import List, Union
import numpy as np
import numpy.typing as npt

import itertools
import autograd.numpy as npa
from autograd import jacobian
from autograd import elementwise_grad as egrad
import scipy


def tf2_get_weights(mlp):
    """_summary_

    Args:
        mlp (_type_): _description_

    Returns:
        _type_: _description_
    """
    return mlp.get_weights()


def pt_get_weights(mlp):
    """_summary_

    Args:
        mlp (_type_): _description_

    Returns:
        _type_: _description_
    """
    params = []
    for param_tensor in mlp.state_dict():
        param = mlp.state_dict()[param_tensor].numpy()
        if "weight" in param_tensor.split("."):
            params.append(param.T)
        else:
            params.append(param)

    return params


def tf2_get_architecture(model):
    """Extracts the arhitecture of tf model (tf2)

    Args:
        model (_type_): _description_
    """
    if model is None:
        raise TypeError("Model cannot be None")
    else:
        architecture = []
        for lnum, lay in enumerate(model.layers):
            if len(lay.weights) != 0:
                architecture.append(lay.input.shape[1])
                if lnum == len(model.layers) - 1:
                    architecture.append(lay.output.shape[1])
    return architecture


def pt_get_architecture(model):
    """Extracts the arhitecture of tf model (tf2)

    Args:
        model (_type_): _description_
    """
    if model is None:
        raise TypeError("Model cannot be None")
    else:
        architecture = []
        for lay in model.state_dict():
            if "weight" in set(lay.split(".")):
                architecture.append(model.state_dict()[lay].size()[1])
                fin_layer = lay

        architecture.append(model.state_dict()[fin_layer].size()[0])
    return architecture


def generate_inside_constraints(name: str, A: npt.NDArray, b: npt.NDArray):
    if A is None:
        raise ValueError("A cannot be empty")
    if b is None:
        raise ValueError("b cannot be empty")
    if len(b.shape) != 2:
        raise ValueError("Shape of b must be N x 1 dimension")
    A_m, A_n = A.shape
    b_n, b_p = b.shape

    if A_m != b_n:
        raise ValueError(
            "First dimension of A (= {}) should be equal to be first dimesnion of b(= {}).".format(
                A_m, b_n
            )
        )
    if b_p != 1:
        raise ValueError("Second dimension of b (={}) should be 1".format(b_p))

    def_string = ""
    add_attr_list = []
    for i in range(A_m):
        single_def_string = ""
        temp_string = []

        for j in range(A_n):
            t_st = "({})*getattr(model, x_l)[i, {}]".format(A[i, j], j)

            temp_string.append(t_st)
        add_attr_string = "setattr(self.model, '{}{}'+str(self.layer_num_next),pyo.Constraint(range(m), rule={}{}))".format(
            name, i, name, i
        )
        add_attr_list.append(add_attr_string)
        return_string = (
            "(" + " + ".join(temp_string) + "  - ({}) <= 0)".format(b[i, 0])
        )
        single_def_string = (
            "def {}{}(model, i):\n\treturn ".format(name, i) + return_string
        )
        def_string = def_string + single_def_string + "\n\n"

    attr_string = "\n".join(add_attr_list)

    def_string += attr_string

    return def_string


def generate_outside_constraints(name: str, A, B):
    if not A:
        raise ValueError("A cannot be empty")
    if not B:
        raise ValueError("B cannot be empty")
    if len(A) != len(B):
        raise ValueError(
            f"Length Mismatch betweeb A and B. Length of A is {len(A)} and Length of B is {len(B)}."
        )

    def_string = ""
    add_attr_list = []
    list_alleq = []
    for a, b in zip(A, B):

        list_eq = []
        iterate_1 = 0

        for a_, b_ in zip(a, b):
            str_eq = ""
            for iterate_2, val in enumerate(a_):

                if val != 0:
                    str_eq += (
                        "("
                        + str(val)
                        + " * "
                        + "getattr(model, x_l)[i"
                        + ", "
                        + str(iterate_2)
                        + "]) + "
                    )

            iterate_1 += 1
            str_eq_whole = ""
            if str_eq != "":
                str_eq_whole = str_eq[0:-2] + " <= " + str(b_[0])
            list_eq.append(str_eq_whole)
        list_alleq.append(list_eq)

    list_alleq = np.array(list_alleq).T

    str_eq_all = "["
    for iterate in range(list_alleq.shape[0]):
        str_eq = "["
        for iterate_2 in range(list_alleq.shape[1]):
            if list_alleq[iterate, iterate_2] != "":
                str_eq += list_alleq[iterate, iterate_2] + ","

        str_eq = str_eq[0:-1] + "]"

        if iterate != list_alleq.shape[0] - 1:
            str_eq_all += str_eq + ",\n\t"
        else:
            str_eq_all += str_eq + "]"

    single_def_string = (
        "def " + name + "0(model, i):\n\treturn " + str_eq_all + "\n"
    )
    single_def_string += (
        "setattr(self.model, '"
        + name
        + "0' +str(self.layer_num_next), pyg.Disjunction(range(m), rule={}{}))".format(
            name, str(0)
        )
    )

    return single_def_string


@dataclass
##################################
# TODO (Tanmay): ConstraintClass should be modified to also receive constraints of the form Cy = d.
# This is what we claimed in the paper. Also, for now let's keep "inside" as the default type of constraint.
# So, ConstraintClass should be like this: ConstraintClass(A,b,C,d) or ConstraintClass(A,b) or ConstraintClass(C,d)
class ConstraintsClass:
    constraint_type: str
    A: Union[List[npt.NDArray], npt.NDArray]
    B: Union[List[npt.NDArray], npt.NDArray]


def generate_output_constraints(constraint):
    outside_count = 0
    inside_count = 0
    out_string = ""
    in_string = ""
    for i in constraint:

        # print(inside_count, outside_count)
        if i.constraint_type == "outside":
            temp_string = generate_outside_constraints(
                i.constraint_type + str(outside_count), i.A, i.B
            )
            # print(temp_string)
            # print("*************************************")
            out_string = out_string + temp_string + "\n\n"
            outside_count = outside_count + 1
        elif i.constraint_type == "inside":
            temp_string = generate_inside_constraints(
                i.constraint_type + str(outside_count), i.A, i.B
            )
            # print(temp_string)
            # print("*************************************")
            in_string = in_string + temp_string + "\n\n"
            inside_count = inside_count + 1

    fin = out_string + "\n\n" + in_string
    return fin


##################################


def give_mse_error(data1: npt.NDArray, data2: npt.NDArray):
    """return the mean square error of data1-data2 samples

    Args:
        data1 (ndarray): predicted targets
        data2 (ndarray): original targets

    Returns:
        float: mse error
    """
    if data1 is None or data2 is None:
        raise TypeError("Data cannot be None")
    row_1, col_1 = np.array(data1).shape
    row_2, col_2 = np.array(data2).shape
    if row_1 != row_2 or col_1 != col_2:
        raise ValueError(
            f"Possible row mismatch. Data 1 has shape {np.array(data1).shape} and Data 2 has shape {np.array(data2).shape}"
        )

    _squared_sum = 0
    for i in range(row_1):
        for j in range(col_1):
            _squared_sum += (data1[i, j] - data2[i, j]) ** 2

    return _squared_sum / row_1


# TODO: Add the below bound stat tracker
class BoundStatTracker:
    def __init__(self, architechture: List[int]):
        self.architechture = architechture
        self.bound_stats = {}
        for i in range(len(architechture)):
            self.bound_stats[i] = {}
            self.bound_stats[i]["max_ub"] = np.zeros(architechture[i])
            self.bound_stats[i]["min_lb"] = np.zeros(architechture[i])
            self.bound_stats[i]["max_lb"] = -np.inf * np.ones(architechture[i])
            self.bound_stats[i]["min_ub"] = np.inf * np.ones(architechture[i])
            self.bound_stats[i]["avg_ub"] = np.zeros(architechture[i])
            self.bound_stats[i]["avg_lb"] = np.zeros(architechture[i])
            self.bound_stats[i]["stably_active_nodes"] = np.zeros(
                architechture[i]
            )
            self.bound_stats[i]["stably_inactive_nodes"] = np.zeros(
                architechture[i]
            )
            self.bound_stats[i]["node_count"] = np.zeros(architechture[i])

    def update_stats(self, lb, ub, layer_num, node_num):
        if self.bound_stats[layer_num]["node_count"][node_num] != 0:
            self.bound_stats[layer_num]["max_ub"][node_num] = max(
                self.bound_stats[layer_num]["max_ub"][node_num], ub
            )
        else:
            self.bound_stats[layer_num]["max_ub"][node_num] = ub
        if self.bound_stats[layer_num]["node_count"][node_num] != 0:
            self.bound_stats[layer_num]["min_lb"][node_num] = min(
                self.bound_stats[layer_num]["min_lb"][node_num], lb
            )
        else:
            self.bound_stats[layer_num]["min_lb"][node_num] = lb
        if self.bound_stats[layer_num]["node_count"][node_num] != 0:
            self.bound_stats[layer_num]["max_lb"][node_num] = max(
                self.bound_stats[layer_num]["max_lb"][node_num], lb
            )
        else:
            self.bound_stats[layer_num]["max_lb"][node_num] = lb
        if self.bound_stats[layer_num]["node_count"][node_num] != 0:
            self.bound_stats[layer_num]["min_ub"][node_num] = min(
                self.bound_stats[layer_num]["min_ub"][node_num], ub
            )
        else:
            self.bound_stats[layer_num]["min_ub"][node_num] = ub

        if self.bound_stats[layer_num]["node_count"][node_num] != 0:
            self.bound_stats[layer_num]["avg_ub"][node_num] = (
                self.bound_stats[layer_num]["avg_ub"][node_num]
                * self.bound_stats[layer_num]["node_count"][node_num]
                + ub
            ) / (self.bound_stats[layer_num]["node_count"][node_num] + 1)
            self.bound_stats[layer_num]["avg_lb"][node_num] = (
                self.bound_stats[layer_num]["avg_lb"][node_num]
                * self.bound_stats[layer_num]["node_count"][node_num]
                + lb
            ) / (self.bound_stats[layer_num]["node_count"][node_num] + 1)
        else:
            self.bound_stats[layer_num]["avg_ub"][node_num] = ub
            self.bound_stats[layer_num]["avg_lb"][node_num] = lb
        self.bound_stats[layer_num]["node_count"][node_num] += 1
        if lb >= 0:
            self.bound_stats[layer_num]["stably_active_nodes"][node_num] += 1
        if ub <= 0:
            self.bound_stats[layer_num]["stably_inactive_nodes"][node_num] += 1

    def print_stats(self, layer_num):
        for node in range(self.architechture[layer_num]):
            if self.bound_stats[layer_num]["node_count"][node] != 0:
                print(f"Layer {layer_num}, Node {node}")
                # print stats for each node per line
                print(
                    f"    Max ub: {self.bound_stats[layer_num]['max_ub'][node]}"
                )
                print(
                    f"    Min lb: {self.bound_stats[layer_num]['min_lb'][node]}"
                )
                print(
                    f"    Max lb: {self.bound_stats[layer_num]['max_lb'][node]}"
                )
                print(
                    f"    Min ub: {self.bound_stats[layer_num]['min_ub'][node]}"
                )
                print(
                    f"    Avg ub: {self.bound_stats[layer_num]['avg_ub'][node]}"
                )
                print(
                    f"    Avg lb: {self.bound_stats[layer_num]['avg_lb'][node]}"
                )
                if layer_num != len(self.architechture) - 1:
                    print(
                        f"    Stably active nodes: {self.bound_stats[layer_num]['stably_active_nodes'][node]}/{self.bound_stats[layer_num]['node_count'][node]}"
                    )
                    print(
                        f"    Stably inactive nodes: {self.bound_stats[layer_num]['stably_inactive_nodes'][node]}/{self.bound_stats[layer_num]['node_count'][node]}"
                    )
                print(f"_______")
                print(" ")


###############################################################################
# TODO:
def neural_return_weights_pert(params, architechture, layer):

    idx = 0
    num_layers = architechture.shape[0] - 1
    weights = []
    bias = []
    for l in range(num_layers):
        inp_dim = architechture[l]
        out_dim = architechture[l + 1]
        W = params[idx : idx + inp_dim * out_dim].reshape((inp_dim, out_dim))
        b = params[idx + inp_dim * out_dim : idx + inp_dim * out_dim + out_dim]
        idx = idx + inp_dim * out_dim + out_dim
        weights.append(W)
        bias.append(b)
    return list(np.where(bias[layer] != 0)[0])


def get_weight_range(arch, layer):
    idx = 0
    num_layers = arch.shape[0] - 1
    for l in range(0, layer + 1):
        idx_range = [idx]
        inp_dim = arch[l]
        out_dim = arch[l + 1]
        idx = idx + inp_dim * out_dim + out_dim
        idx_range.append(idx)

    return idx_range


def get_vec_idx_sparse(arch, layer, columns):
    w_mask = np.zeros((arch[layer], arch[layer + 1]))
    b_mask = np.zeros(arch[layer + 1])
    for c in columns:
        w_mask[:, c] = 1
        b_mask[c] = 1

    w_mask = w_mask.flatten()
    b_mask = b_mask.flatten()
    agg_mask = np.hstack((w_mask, b_mask))
    idx_list = list(np.where(agg_mask == 1)[0])
    return idx_list


def sparse_eigenvector_reduction(H_dist, arch, layer, num_non_sparse):
    # get layer indices
    idx_range = get_weight_range(arch, layer)
    entry_list = list(range(idx_range[0], idx_range[1]))
    column_list = list(range(arch[layer + 1]))

    # test all num_non_sparse combinations of weights
    # and check which one maximizes w.T*H*w
    max_cost = 0
    iteration = 1
    all_iters = int(
        scipy.special.factorial(arch[layer + 1])
        / (
            scipy.special.factorial(num_non_sparse)
            * scipy.special.factorial(arch[layer + 1] - num_non_sparse)
        )
    )
    costs = []
    all_vec = []
    for L in range(num_non_sparse, num_non_sparse + 1):
        for subset in itertools.combinations(column_list, L):
            column_idx = list(subset)
            idx_list = get_vec_idx_sparse(arch, layer, column_idx)
            sparse_idx = [entry_list[i] for i in idx_list]
            H_reduced = H_dist[sparse_idx][:, sparse_idx]

            #             if not isPD(H_reduced):
            #                 H_reduced = nearestPD(H_reduced)

            ei, vi = np.linalg.eig(H_reduced)
            vec_reduced = vi[:, 0]
            vec = np.zeros(H_dist.shape[0])
            idx = 0
            for el in sparse_idx:
                vec[el] = vec_reduced[idx]
                idx += 1
            cost = np.matmul(vec, np.matmul(H_dist, vec))
            costs.append(cost)
            all_vec.append(vec)
            print(
                f"iteration: {iteration}/{all_iters}, eval nodes: {column_idx}, cost: {cost}"
            )
            if cost >= max_cost:
                selected_vec = vec
                max_cost = cost
            iteration += 1
    print(f"max cost: {max_cost}")
    return selected_vec, costs, all_vec


def neural_net_predict(params, inputs, architechture):

    idx = 0
    num_layers = architechture.shape[0] - 1
    for l in range(num_layers):
        inp_dim = architechture[l]
        out_dim = architechture[l + 1]
        W = params[idx : idx + inp_dim * out_dim]
        b = params[idx + inp_dim * out_dim : idx + inp_dim * out_dim + out_dim]
        outputs = npa.dot(inputs, W.reshape((inp_dim, out_dim))) + b
        inputs = npa.maximum(
            npa.zeros((outputs.shape[0], outputs.shape[1])), outputs
        )
        idx = idx + inp_dim * out_dim + out_dim

    return outputs


def __soft_plus(x, t=10.0):  # (1/t) * Ln(1+exp(t*x))
    return (1 / t) * npa.log(1 + npa.exp(t * x))


def get_sensitive_nodes(
    model, layer_to_repair, x_train, num_sparse_nodes, A, b
):

    A = npa.array(A)
    b = npa.array(b)

    def objective(params):
        y_pred = neural_net_predict(params, x_train, architecture)
        const = -(npa.matmul(A, y_pred[:, 0 : A[0].shape[0]].T) - b)
        loss = npa.sum(
            # npa.maximum(np.zeros((const.shape[0], const.shape[1])), const)
            __soft_plus(const, 20)
            # / const.shape[0]
        )
        return loss

    architecture = np.array(tf2_get_architecture(model))
    w_b = tf2_get_weights(model)
    ws = []
    for w in w_b:
        ws = ws + list(w.flatten())
    init_params = npa.array(ws)
    eigenvector, _, _ = sparse_eigenvector_reduction(
        jacobian(egrad(objective))(init_params),
        architecture,
        layer_to_repair - 1,
        num_sparse_nodes,
    )

    return neural_return_weights_pert(
        eigenvector, architecture, layer_to_repair - 1
    )


###############################################################################
