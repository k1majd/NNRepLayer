import os
import numpy as np
import numpy.typing as npt
from tensorflow import keras
import itertools
import autograd.numpy as npa
from autograd import jacobian
from autograd import elementwise_grad as egrad
import scipy
from shapely.affinity import scale
from nnreplayer.utils.utils import tf2_get_architecture, tf2_get_weights
import matplotlib.pyplot as plt
import csv

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
    num_w_b = 0
    for i in range(len(arch) - 1):
        num_w_b += arch[i] * arch[i + 1] + arch[i + 1]

    idx_range = [0, num_w_b]
    entry_list = list(range(idx_range[0], idx_range[1]))
    column_list = list(range(arch[layer + 1]))

    # test all num_non_sparse combinations of weights
    # and check which one maximizes w.T*H*w
    max_cost = 0
    iteration = 1
    all_iters = int(
        scipy.special.factorial(num_w_b)
        / (
            scipy.special.factorial(num_non_sparse)
            * scipy.special.factorial(num_w_b - num_non_sparse)
        )
    )
    costs = []
    all_vec = []
    for L in range(num_non_sparse, num_non_sparse + 1):
        for subset in itertools.combinations(entry_list, L):
            sparse_idx = list(subset)
            # idx_list = get_vec_idx_sparse(arch, layer, column_idx)
            # sparse_idx = [entry_list[i] for i in idx_list]
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
                f"iteration: {iteration}/{all_iters}, eval weights: {sparse_idx}, cost: {cost}"
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
    objective, init_params, architecture, num_sparse_nodes, A, b
):

    A = npa.array(A)
    b = npa.array(b)

    layer_to_repair = 1
    hessian_loss = jacobian(egrad(objective))(init_params)
    eigenvector, cost_vec, all_vec = sparse_eigenvector_reduction(
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
    return list(np.where(eigenvector != 0)[0]), eigenvector, hessian_loss


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    return data


def generateDataWindow(window_size):
    Dfem = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_1.csv"
    )
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_2.csv"
    )
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_3.csv"
    )
    n = 20364
    Dankle = np.subtract(Dtib[:n, 1], Dfut[:n, 1])
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    n_train = 18200
    # n_train = 500
    train_observation = np.array([]).reshape(0, 4 * window_size)
    test_observation = np.array([]).reshape(0, 4 * window_size)
    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        train_observation = np.concatenate(
            (train_observation, temp_obs), axis=0
        )
    train_controls = controls[window_size : n_train + window_size].reshape(
        -1, 1
    )
    for i in range(n_train, n - window_size):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size :].reshape(-1, 1)
    return train_observation, train_controls, test_observation, test_controls


def main(given_comp):
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)
    rnd_pts = np.random.choice(test_obs.shape[0], 100)
    x_train = test_obs[rnd_pts]
    y_train = test_ctrls[rnd_pts]

    # ctrl_model_orig, callback, architecture = buildModelWindow(train_obs.shape)
    model_orig = keras.models.load_model(
        os.path.dirname(os.path.realpath(__file__))
        + "/models/model_orig/original_model"
    )
    # ctrl_model_orig.load_weights('models/model1')
    # ctrl_model_orig.fit(
    #     train_obs,
    #     train_ctrls,
    #     validation_data=(test_obs, test_ctrls),
    #     batch_size=8,
    #     epochs=20,
    #     use_multiprocessing=True,
    #     verbose=1,
    #     shuffle=False,
    #     callbacks=callback,
    # )
    # plotTestData(ctrl_model_orig, train_obs, train_ctrls, test_obs, test_ctrls)

    bound_upper = 10
    bound_lower = 30

    A = np.array([[1]])
    b = np.array([[bound_upper]])

    # constraint cost
    def objective(params):
        y_pred = neural_net_predict(params, x_train, architecture)
        const = -(npa.matmul(A, y_pred[:, 0 : A[0].shape[0]].T) - b)
        loss = npa.sum(
            # np.maximum(np.zeros(const.shape[0]), const)
            __soft_plus(const, 20)
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
            __soft_plus(const, 20)
            / const.shape[0]
        )
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
    repair_indices, eig, hessian = get_sensitive_nodes(
        objective, init_params, architecture, 2, A, b
    )
    print(f"repair indices: {repair_indices}")
    if len(given_comp) == 0:
        # plot 1:
        w1 = np.linspace(
            init_params[repair_indices[0]] - 1,
            init_params[repair_indices[0]] + 1,
            100,
        )
        w2 = np.linspace(
            init_params[repair_indices[1]] - 1,
            init_params[repair_indices[1]] + 1,
            100,
        )
        print(
            f"eigenvector: {[eig[repair_indices[0]], eig[repair_indices[1]]]}"
        )
        w0 = [init_params[repair_indices[0]], init_params[repair_indices[1]]]
        print(w0)
        W1, W2 = np.meshgrid(w1, w2)
        obj = np.zeros((W1.shape[0], W1.shape[1]))

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                init_params[repair_indices[0]] = W1[i, j]
                init_params[repair_indices[1]] = W2[i, j]
                obj[i, j] = objective2(init_params)

        plt.contour(W1, W2, obj, 1000, levels=[0])
        plt.contourf(W1, W2, obj, 1000, cmap="RdGy")
        plt.colorbar()
        plt.plot(
            [w1[0], w1[-1]],
            [
                (eig[repair_indices[1]] / eig[repair_indices[0]])
                * (w1[0] - w0[0])
                + w0[1],
                (eig[repair_indices[1]] / eig[repair_indices[0]])
                * (w1[-1] - w0[0])
                + w0[1],
            ],
            label="eigenvector direction",
        )
        plt.scatter(
            w0[0], w0[1], s=80, marker="+", label="original weight value"
        )
        plt.xlabel(f"w{repair_indices[0]}")
        plt.ylabel(f"w{repair_indices[1]}")
        plt.title(
            f"contour plot of constraint surface - highest sensitive weights"
        )
        plt.legend()
        plt.show()
    else:
        repair_indices = given_comp
        _, ev2 = np.linalg.eig(hessian[repair_indices][:, repair_indices])
        eig = ev2[:, 0]
        # plot 2:
        w1 = np.linspace(
            init_params[repair_indices[0]] - 1,
            init_params[repair_indices[0]] + 1,
            100,
        )
        w2 = np.linspace(
            init_params[repair_indices[1]] - 1,
            init_params[repair_indices[1]] + 1,
            100,
        )
        print(f"eigenvector: {[eig[0], eig[1]]}")
        w0 = [init_params[repair_indices[0]], init_params[repair_indices[1]]]
        print(w0)
        W1, W2 = np.meshgrid(w1, w2)
        obj = np.zeros((W1.shape[0], W1.shape[1]))

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                init_params[repair_indices[0]] = W1[i, j]
                init_params[repair_indices[1]] = W2[i, j]
                obj[i, j] = objective2(init_params)

        # plt.contour(W1, W2, obj, 1000, levels=[0])
        plt.contourf(W1, W2, obj, 1000, cmap="RdGy")
        plt.colorbar()
        plt.plot(
            [w1[0], w1[-1]],
            [
                (eig[1] / eig[0]) * (w1[0] - w0[0]) + w0[1],
                (eig[1] / eig[0]) * (w1[-1] - w0[0]) + w0[1],
            ],
            label="eigenvector direction",
        )
        plt.scatter(
            w0[0], w0[1], s=80, marker="+", label="original weight value"
        )
        plt.xlabel(f"w{repair_indices[0]}")
        plt.ylabel(f"w{repair_indices[1]}")
        plt.title(
            f"contour plot of constraint surface - highest sensitive weights"
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main([])
