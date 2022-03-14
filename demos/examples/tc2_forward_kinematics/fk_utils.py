import os
import pickle
import numpy as np
from shapely.geometry import Polygon, Point
from sklearn.model_selection import train_test_split
from quadprog import solve_qp
from matplotlib import pyplot as plt
import matplotlib as mpl

## batch creation
class Batch(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, X_col, Y_col, batch_size_val):
        """_summary_

        Args:
            X_col (_type_): _description_
            Y_col (_type_): _description_
            batch_size_val (_type_): _description_
        """
        self.X = X_col
        self.Y = Y_col
        self.size = X_col.shape[0]
        self.train_size = batch_size_val
        self.test_size = self.size - batch_size_val

    def get_batch(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        values = range(self.size)
        train_dataset, test_dataset = train_test_split(
            values, train_size=self.train_size, test_size=self.test_size
        )
        return (
            self.X[train_dataset, :],
            self.Y[train_dataset, :],
            self.X[test_dataset, :],
            self.Y[test_dataset, :],
        )


## reshape input
def reshape_cos_sin(inp):
    """_summary_

    Args:
        inp (_type_): _description_

    Returns:
        _type_: _description_
    """
    row = inp.shape[0]
    col = inp.shape[1]
    x = np.zeros((row, 2 * col))
    for i in range(row):
        for j in range(col):
            x[i, 2 * j] = np.cos(inp[i, j])
            x[i, 2 * j + 1] = np.sin(inp[i, j])
    return x


def original_data_loader():
    """_summary_

    Raises:
        ImportError: error if the data is does not exist in the designated location

    Returns:
        list[ndarray]: a list of [x_train, y_train, x_test, y_test]
    """
    direc = os.path.dirname(os.path.realpath(__file__))
    path_read = direc + "/tc2/original_net"
    if not os.path.exists(path_read + "/data/input_output_data_tc2.pickle"):
        raise ImportError(
            "path {path_read}/data/input_output_data_tc2.pickle does not exist!"
        )
    with open(path_read + "/data/input_output_data_tc2.pickle", "rb") as data:
        dataset = pickle.load(data)
    return dataset[0], dataset[1], dataset[2], dataset[3]


def label_output_inside(
    inp_data,
    out_data,
    A=np.array([[1.0, 0.0, 0.0, 0.0]]),
    b=0.5,
    bound_error=0.1,
    mode="finetune",
):
    # pylint: disable=invalid-name
    """Project the target points that lie outside of the contrained convex set.
    Outputs either all target data points(retrain) or just projected points (finetune).

    Args:
        poly_const (_type_): _description_
        out_data (_type_): _description_
        bound_error (float, optional): distance of newly labeled points to the boudnary of set.
        mode (str, optional): Hand-labeling mode = "finetune", "retrain". Defaults to "finetune".

    Returns:
        _type_: _description_
    """
    # get the coordinates of the exterior points of the polytope
    out_data_new = []
    inp_data_new = []
    P = np.diag(np.ones(3))
    for i in range(out_data.shape[0]):
        if not np.matmul(A, out_data[i, 0:3]) <= b:
            sol, _, _, _, _, _ = solve_qp(
                P, out_data[i, 0:3], -A.T, -b, meq=0, factorized=True
            )
            dist = np.linalg.norm(out_data[i, 0:3] - sol)
            sol = (bound_error / dist) * (sol - out_data[i, 0:3]) + sol
            out_data_new.append(np.append(sol, [1]))
            inp_data_new.append(inp_data[i, :])
        else:
            if mode == "retrain":
                out_data_new.append(out_data[i, :])
                inp_data_new.append(inp_data[i, :])

    return np.array(inp_data_new), np.array(out_data_new)


## generate data samples
def data_generate(num_pts, unif2edge=0.75, edge_scale=0.7):
    """_summary_

    Args:
        num_pts (int): numebr of data points
        unif2edge (float, optional): #samples taken uniformly/ #samples on edge. Defaults to 0.75.
        edge_scale (float, optional): edge buffer size. Defaults to 0.7.

    Returns:
        _type_: _description_
    """

    num_pts_unif = int(unif2edge * num_pts)
    num_pts_edge = num_pts - num_pts_unif

    my_data = np.genfromtxt(
        os.path.dirname(os.path.realpath(__file__)) + "/raw_dataset.csv",
        delimiter=",",
    )
    x1 = my_data[0:, 0:6]
    y_raw = my_data[0:, 6:]
    x_raw = reshape_cos_sin(x1)  # pass input angles through sin cos filter

    y_mean = (
        np.sum(y_raw, 0) / np.sum(y_raw, 0).flatten()[-1]
    )  # the mean point of output
    y_dist = np.array(
        [
            np.linalg.norm(y_raw[i, 0:3] - y_mean.flatten()[0:3])
            for i in range(y_raw.shape[0])
        ]
    )
    y_dist = np.sort(y_dist, axis=None)
    y_radii = np.sum(y_dist[-10:]) / 10  # avg radius of output ball

    ## randomly select data samples inside the output ball
    random_indices = np.random.choice(
        x_raw.shape[0], size=num_pts_unif, replace=False
    )
    x = x_raw[random_indices, :]
    y = y_raw[random_indices, :]
    x_raw = np.delete(x_raw, random_indices, axis=0)
    y_raw = np.delete(y_raw, random_indices, axis=0)

    ## randomly select data samples on the edge of output ball
    counter = 0
    while counter != num_pts_edge:
        indx = np.random.randint(0, x_raw.shape[0])
        if (
            np.linalg.norm(y_raw[indx, 0:3] - y_mean.flatten()[0:3])
            >= edge_scale * y_radii
        ):
            x = np.vstack((x, x_raw[indx, :]))
            y = np.vstack((y, y_raw[indx, :]))
            x_raw = np.delete(x_raw, indx, axis=0)
            y_raw = np.delete(y_raw, indx, axis=0)
            counter += 1

    return x, y


def model_eval(model_new, model_orig, path_read, constraints):
    # pylint: disable=trailing-whitespace
    # pylint: disable=too-many-locals
    """_summary_

    Args:
        model_new (tf): new model in tensorflow
        model_orig (tf): original model in tensorflow
        path_read (str): path to read stat data
        poly_const (obj): constraint polytope._read Shapely documentation

    Raises:
        ImportError: error if path_read does not exist

    Returns:
        list[float]: a list of [loss for inside test samples,
                                loss for outside test samples,
                                #corrected samples/#train buggy samples,
                                #corrected samples/#test buggy samples,
                                L1 weight error,
                                L2 weight error,
                                Linfty weight error]
    """
    # load eval dataset
    if not (
        os.path.exists(
            path_read + "/data/input_output_data_inside_train_tc2.pickle"
        )
        or os.path.exists(
            path_read + "/data/input_output_data_outside_train_tc2.pickle"
        )
        or os.path.exists(
            path_read + "/data/input_output_data_inside_test_tc2.pickle"
        )
        or os.path.exists(
            path_read + "/data/input_output_data_outside_test_tc2.pickle"
        )
    ):
        raise ImportError(
            "inside-outside datasets for test and train should be generated first in path {path_read}/data/"
        )
    # with open(
    #     path_read + "/data/input_output_data_inside_train_tc2.pickle", "rb"
    # ) as data:
    #     train_inside = pickle.load(data)
    with open(
        path_read + "/data/input_output_data_outside_train_tc2.pickle", "rb"
    ) as data:
        train_out = pickle.load(data)
    with open(
        path_read + "/data/input_output_data_inside_test_tc2.pickle", "rb"
    ) as data:
        test_in = pickle.load(data)
    with open(
        path_read + "/data/input_output_data_outside_test_tc2.pickle", "rb"
    ) as data:
        test_out = pickle.load(data)

    # accuracy on outside and inside models
    loss_test_in = model_new.evaluate(test_in[0], test_in[1], verbose=0)[0]
    loss_test_out = model_new.evaluate(test_out[0], test_out[1], verbose=0)[0]

    # new model accuracy on buggy data - train
    y_train_out_pred = model_new.predict(train_out[0])
    num_buggy_train = train_out[1].shape[0]
    num_corrected_train = num_buggy_train

    for i in range(num_buggy_train):
        if (
            not np.matmul(constraints[0], y_train_out_pred[i])
            <= constraints[1]
        ):
            if (
                np.matmul(constraints[0], y_train_out_pred[i]) - constraints[1]
            ).flatten()[0] > 0.001:
                num_corrected_train -= 1

    acc_buggy_train = num_corrected_train / num_buggy_train

    # new model accuracy on buggy data - test
    y_test_out_pred = model_new.predict(test_out[0])
    num_buggy_test = test_out[1].shape[0]
    num_corrected_test = num_buggy_test

    for i in range(num_buggy_test):
        if not np.matmul(constraints[0], y_test_out_pred[i]) <= constraints[1]:
            if (
                np.matmul(constraints[0], y_train_out_pred[i]) - constraints[1]
            ).flatten()[0] > 0.001:
                num_corrected_test -= 1

    acc_buggy_test = num_corrected_test / num_buggy_test

    # w-w_orig metrics
    w_orig = np.array([])
    w_new = np.array([])
    for w1, w2 in zip(model_orig.get_weights(), model_new.get_weights()):
        w_orig = np.append(w_orig, w1.flatten())
        w_new = np.append(w_new, w2.flatten())

    norm_1 = np.linalg.norm(w_orig - w_new, 1)
    norm_2 = np.linalg.norm(w_orig - w_new)
    norm_inf = np.linalg.norm(w_orig - w_new, np.inf)

    return [
        loss_test_in,
        loss_test_out,
        acc_buggy_train,
        acc_buggy_test,
        norm_1,
        norm_2,
        norm_inf,
    ]


def plot_history(his, include_validation=False):
    """_summary_

    Args:
        his (_type_): _description_
        include_validaation (bool, optional): _description_. Defaults to False.
    """
    print("----------------------")
    print("History Visualization")
    plt.rcParams["text.usetex"] = False
    mpl.style.use("seaborn")

    ## loss plotting
    results_train_loss = his.history["loss"]
    plt.plot(results_train_loss, color="red", label="training loss")
    if include_validation:
        results_valid_loss = his.history["val_loss"]
        plt.plot(results_valid_loss, color="blue", label="validation loss")
    plt.title("Loss Function Output (fine-tuning the last layer)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc="upper left", frameon=False)
    plt.show()

    ## accuracy plotting
    results_train_acc = his.history["accuracy"]
    plt.plot(results_train_acc, color="red", label="training accuracy")
    if include_validation:
        results_valid_acc = his.history["val_accuracy"]
        plt.plot(results_valid_acc, color="blue", label="validation accuracy")
    plt.title("Accuracy Function Output (fine-tuning the last layer)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="upper left", frameon=False)
    plt.show()


def plot_dataset3d(out_dataset, data_labels, title_label="training"):
    """_summary_

    Args:
        polys (_type_): _description_
        out_dataset (_type_): _description_
        label (str, optional): _description_. Defaults to "training".
    """
    print("----------------------")
    print(f"Data samples Visualization ({title_label})")
    plt.rcParams["text.usetex"] = False
    mpl.style.use("seaborn")
    ax = plt.axes(projection="3d")
    colors = ["plum", "tab:blue", "tab:red", "green"]
    for i, data in enumerate(out_dataset):
        ax.scatter3D(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            color=colors[i],
            label=data_labels[i],
        )

    plt.legend(loc="upper left", frameon=False, fontsize=20)
    plt.title(f"Forward Kinematics ({title_label} dataset)", fontsize=20)
    plt.xlabel("x", fontsize=25)
    plt.ylabel("y", fontsize=25)
    plt.show()
