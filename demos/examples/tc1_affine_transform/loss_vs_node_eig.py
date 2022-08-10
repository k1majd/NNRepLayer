import os
import numpy as np
import numpy.typing as npt
from tensorflow import keras
import itertools
import autograd.numpy as npa
from autograd import jacobian
from autograd import elementwise_grad as egrad
import scipy
from csv import writer
from shapely.affinity import scale
from affine_utils import (
    original_data_loader,
    give_polys,
    give_constraints,
)
from nnreplayer.utils.utils import tf2_get_architecture, tf2_get_weights
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")
import numpy as np


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
    idx_range = [0]
    num_layers = arch.shape[0] - 1
    for l in range(0, layer + 1):
        inp_dim = arch[l]
        out_dim = arch[l + 1]
        idx = idx + inp_dim * out_dim + out_dim
    idx_range.append(idx)

    return idx_range


def get_vec_idx_sparse(arch, layer, columns):
    col_list = []
    counter = 0
    for l in range(arch.shape[0] - 1):
        temp = [
            c - counter
            for c in columns
            if (c >= counter and c < counter + arch[l + 1])
        ]
        col_list.append(temp)
        counter += arch[l + 1]

    w_b_mask = []
    for l in range(arch.shape[0] - 1):
        w_temp = np.zeros((arch[l], arch[l + 1]))
        b_temp = np.zeros(arch[l + 1])
        for c in col_list[l]:
            w_temp[:, c] = 1
            b_temp[c] = 1
        w_b_mask.append(w_temp)
        w_b_mask.append(b_temp)

    ws = []
    for w in w_b_mask:
        ws = ws + list(w.flatten())
    mask_params = np.array(ws)

    idx_list = list(np.where(mask_params == 1)[0])
    return idx_list


def sparse_eigenvector_reduction(H_dist, arch, layer, num_non_sparse):
    # get layer indices
    # idx_range = get_weight_range(arch, layer)
    # entry_list = list(range(idx_range[0], idx_range[1]))
    column_list = list(range(np.sum(arch[1:])))

    # test all num_non_sparse combinations of weights
    # and check which one maximizes w.T*H*w
    max_cost = 0
    iteration = 1
    all_iters = int(
        scipy.special.factorial(len(column_list))
        / (
            scipy.special.factorial(num_non_sparse)
            * scipy.special.factorial(len(column_list) - num_non_sparse)
        )
    )
    costs = []
    all_vec = []
    all_idx = []
    for L in range(num_non_sparse, num_non_sparse + 1):
        for subset in itertools.combinations(column_list, L):
            column_idx = list(subset)
            all_idx.append(column_idx)
            sparse_idx = get_vec_idx_sparse(arch, layer, column_idx)
            # sparse_idx = [entry_list[i] for i in idx_list]
            H_reduced = H_dist[sparse_idx][:, sparse_idx]

            #             if not isPD(H_reduced):
            #                 H_reduced = nearestPD(H_reduced)

            ei, vi = np.linalg.eig(H_reduced)
            # vec_reduced = vi[:, 0]
            vec = np.zeros((H_dist.shape[0], vi.shape[0]))

            for c in range(vi.shape[1]):
                idx = 0
                for el in sparse_idx:
                    vec[el, c] = vi[idx, c]
                    idx += 1
            cost = np.matmul(vec[:, 0], np.matmul(H_dist, vec[:, 0]))
            costs.append(cost)
            all_vec.append(vec)
            print(
                f"iteration: {iteration}/{all_iters}, eval nodes: {column_idx}, cost: {cost}"
            )
            if cost >= max_cost:
                selected_vec = vec[:, 0]
                max_cost = cost
                selected_nodes = column_idx
            iteration += 1
    print(f"max cost: {max_cost}, selected nodes: {selected_nodes}")
    return selected_nodes, selected_vec, costs, all_vec, all_idx


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
    objective,
    init_params,
    architecture,
    num_sparse_nodes,
):

    layer_to_repair = 3
    hessian_loss = jacobian(egrad(objective))(init_params)
    (
        selected_nodes,
        eigenvector,
        cost_vec,
        all_vec,
        all_idx,
    ) = sparse_eigenvector_reduction(
        hessian_loss, architecture, layer_to_repair - 1, num_sparse_nodes
    )
    # max_16_indices = list(np.argsort(cost_vec))
    # repair_indices = []
    # for id in max_16_indices:
    #     repair_indices.append(
    #         neural_return_weights_pert(
    #             all_vec[id], architecture, layer_to_repair - 1
    #         )
    #     )
    return selected_nodes, cost_vec, all_vec, all_idx


def main(given_comp):
    direc = os.path.dirname(os.path.realpath(__file__))
    path_read = direc + "/tc1/original_net"
    # check_log_directories(path_read, path_write, layer_to_repair)

    # load model
    model_orig = keras.models.load_model(path_read + "/model")

    # load dataset and constraints
    x_train, y_train, x_test, y_test = original_data_loader()
    poly_orig, poly_trans, poly_const = give_polys()
    A, b = give_constraints(
        scale(poly_const, xfact=0.98, yfact=0.98, origin="center")
    )

    # constraint cost
    def objective(params):
        y_pred = neural_net_predict(params, x_train, architecture)
        const = -(npa.matmul(A, y_pred[:, 0 : A[0].shape[0]].T) - b)
        loss = npa.sum(
            # np.maximum(np.zeros(const.shape[0]), const)
            __soft_plus(const, 100)
            / const.shape[0]
        )
        # loss = 0.0
        # for i in range(const.shape[0]):
        #     for j in range(const.shape[1]):
        #         if const[i, j] > 0:
        #             loss += const[i, j]

        return loss

    def objective2(params):
        y_pred = neural_net_predict(params, x_train, architecture)
        const = npa.matmul(A, y_pred[:, 0 : A[0].shape[0]].T) - b
        loss = npa.sum(
            # np.maximum(np.zeros(const.shape[0]), const)
            __soft_plus(const, 30)
            / const.shape[0]
        )
        # loss = 0.0
        # for i in range(const.shape[0]):
        #     for j in range(const.shape[1]):
        #         if const[i, j] > 0:
        #             loss += const[i, j]

        return loss

    def objective4(params):
        y_pred = neural_net_predict(params, x_train, architecture)
        const = npa.matmul(A, y_pred[:, 0 : A[0].shape[0]].T) - b
        loss = npa.sum(
            # np.maximum(np.zeros(const.shape[0]), const)
            __soft_plus(const, 50)
            / const.shape[0]
        )
        # loss = 0.0
        # for i in range(const.shape[0]):
        #     for j in range(const.shape[1]):
        #         if const[i, j] > 0:
        #             loss += const[i, j]

        return loss

    def objective3(params):
        y_pred = neural_net_predict(params, x_train, architecture)
        loss = 0
        vertices = [
            [npa.array([2.5, 0.767]), npa.array([0.767, 2.5])],
            [npa.array([4.233, 2.5]), npa.array([2.5, 0.767])],
            [npa.array([0.767, 2.5]), npa.array([2.5, 4.233])],
            [npa.array([2.5, 4.233]), npa.array([4.233, 2.5])],
        ]
        for i in range(y_pred.shape[0]):
            point = y_pred[i, 0 : A[0].shape[0]]
            temp = npa.matmul(A, point) - b.T
            if np.any(temp > 0):
                locs = np.where(temp[0] > 0)[0]
                max_dist = npa.float64(-1.0)
                for id in list(locs):
                    p1 = vertices[id][0]
                    p2 = vertices[id][1]
                    r = npa.dot(p2 - p1, point - p1)
                    r /= npa.linalg.norm(p2 - p1) ** 2
                    if r < 0:
                        dist = npa.linalg.norm(point - p1)
                    elif r > 1:
                        dist = npa.linalg.norm(p2 - point)
                    else:
                        dist = npa.sqrt(
                            npa.linalg.norm(point - p1) ** 2
                            - (r * npa.linalg.norm(p2 - p1)) ** 2
                        )
                    # loss += dist
                    max_dist = npa.maximum(max_dist, dist)
                # print(max_dist._value._value)
                loss += max_dist

        # loss = 0.0
        # for i in range(const.shape[0]):
        #     for j in range(const.shape[1]):
        #         if const[i, j] > 0:
        #             loss += const[i, j]

        return loss

    architecture = np.array(tf2_get_architecture(model_orig))
    w_b = tf2_get_weights(model_orig)
    ws = []
    for w in w_b:
        ws = ws + list(w.flatten())
    init_params = npa.array(ws)

    # get sensitive nodes
    repair_indices, cost_vec, all_vec, all_idx = get_sensitive_nodes(
        objective3, init_params, architecture, 2
    )
    # repair_indices, cost_vec, all_vec = get_sensitive_nodes(
    #     objective2, init_params, architecture, 1
    # )
    # get the indices of the two maximum values of the cost vector
    max_indices = list(np.argsort(cost_vec))
    max_indices = max_indices[-2:]
    max_indices = [4, 8]

    vec1 = all_vec[max_indices][:, 0]
    vec2 = all_vec[max_indices][:, 1]

    w1_pert = np.linspace(
        -0.5,
        0.5,
        300,
    )
    w2_pert = np.linspace(
        -0.5,
        0.5,
        300,
    )
    W1, W2 = np.meshgrid(w1_pert, w2_pert)
    obj = np.zeros((W1.shape[0], W1.shape[1]))

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            w_traget = W1[i, j] * vec1 + W2[i, j] * vec2 + init_params
            # init_params[repair_indices[0]] = W1[i, j]
            # init_params[repair_indices[1]] = W2[i, j]
            obj[i, j] = objective4(w_traget)
            # print(obj[i, j])
    # horizental subplots
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 1, 1)
    ax.contourf(W1, W2, obj, 2000, cmap="RdGy", vmin=obj.min(), vmax=obj.max())
    # ax.contourf(W1, W2, obj, 2000, cmap="RdGy")
    contours = ax.contour(
        W1, W2, obj, 2000, colors="#FFD700", levels=[0.25e-5]
    )
    ax.clabel(contours, inline=True, fmt="%1.1f")
    ax.set_title("Objective")
    ax.set_xlabel(f"ev1_w{all_idx[max_indices][0]}_w{all_idx[max_indices][1]}")
    ax.set_ylabel(f"ev2_w{all_idx[max_indices][0]}_w{all_idx[max_indices][1]}")
    # add colorbar
    cbar = fig.colorbar(ax.contourf(W1, W2, obj, cmap="RdGy"))
    cbar.ax.set_ylabel("Objective")

    # axs[1] to have 3d projection
    ax = fig.add_subplot(2, 1, 2, projection="3d")
    contours = ax.contour(
        W1, W2, obj, 2000, colors="#FFD700", levels=[0.25e-5]
    )
    ax.clabel(contours, inline=True, fmt="%1.1f")
    ax.contour3D(W1, W2, obj, 50, cmap="RdGy")
    ax.set_xlabel(f"ev1_w{all_idx[max_indices][0]}_w{all_idx[max_indices][1]}")
    ax.set_ylabel(f"ev2_w{all_idx[max_indices][0]}_w{all_idx[max_indices][1]}")
    # plt.contour(W1, W2, obj, 2000, colors="#FFD700", levels=[0])
    # plt.contourf(W1, W2, obj, 2000, cmap="RdGy")
    # # plt.colorbar()
    # ax = plt.axes(projection="3d")
    # ax.contour3D(W1, W2, obj, 50, cmap="RdGy")
    # ax.contour(W1, W2, obj, 2000, colors="#FFD700", levels=[0])
    # ax.colorbar()
    # plt.xlabel(f"w{max_indices[0]}")
    # plt.ylabel(f"w{max_indices[1]}")
    # plt.title(
    #     f"contour plot of constraint surface - highest sensitive weights"
    # )
    # plt.legend()
    plt.show()

    print(f"repair indices: {repair_indices}")
    # pylint: disable=unspecified-encoding
    with open(
        f"costs2.csv",
        "a+",
        newline="",
    ) as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list(cost_vec))
    print("saved: stats")

    print(f"repair indices: {repair_indices}")


if __name__ == "__main__":
    main([])
