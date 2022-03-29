from cProfile import label
import os
import numpy as np
import shapely.affinity as affinity
from sklearn.model_selection import train_test_split
from quadprog import solve_qp
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt, units
import matplotlib as mpl
import pickle
import colorsys


def give_polys():
    """_summary_"""
    ## affine transformation matrices
    translate1 = np.array(
        [[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]]
    )  # translation matrix 1
    translate2 = np.array(
        [[1, 0, -2.5], [0, 1, -2.5], [0, 0, 1]]
    )  # translation matrix 2
    rotate = np.array(
        [
            [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
            [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
            [0, 0, 1],
        ]
    )  # rotation matrix

    ## original, transformed, and constraint Polygons
    poly_orig = Polygon([(1, 1), (4, 1), (4, 4), (1, 4)])
    poly_trans = Polygon(
        [(2.5, 4.621), (4.624, 2.5), (2.5, 0.3787), (0.3787, 2.5)]
    )
    inp_const_vertices = np.array(
        [[1.25, 3.75, 3.75, 1.25], [1.25, 1.25, 3.75, 3.75], [1, 1, 1, 1]]
    )  # contraint vertices in input space
    out_const_vertices = np.matmul(
        np.matmul(np.matmul(translate1, rotate), translate2),
        inp_const_vertices,
    )  # constraint vertices in output space
    poly_const = Polygon(
        [
            (out_const_vertices[0, 0], out_const_vertices[1, 0]),
            (out_const_vertices[0, 1], out_const_vertices[1, 1]),
            (out_const_vertices[0, 2], out_const_vertices[1, 2]),
            (out_const_vertices[0, 3], out_const_vertices[1, 3]),
        ]
    )
    return poly_orig, poly_trans, poly_const


def give_constraints(poly):
    """_summary_

    Args:
        poly (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get the coordinates of the exterior points of the polytope
    ex_points = np.array(poly.exterior.coords)

    # get A and b matrices
    hull = ConvexHull(ex_points)
    eqs = np.array(hull.equations)
    A = eqs[0 : eqs.shape[0], 0 : eqs.shape[1] - 1]
    b = -eqs[0 : eqs.shape[0], -1]

    b = np.array([b]).T

    return A, b


def original_data_loader():
    """_summary_

    Raises:
        ImportError: error if the data is does not exist in the designated location

    Returns:
        list[ndarray]: a list of [x_train, y_train, x_test, y_test]
    """
    direc = os.path.dirname(os.path.realpath(__file__))
    path_read = direc + "/tc1/original_net"
    if not os.path.exists(path_read + "/data/input_output_data_tc1.pickle"):
        raise ImportError(
            "path {path_read}/data/input_output_data_tc1.pickle does not exist!"
        )
    with open(path_read + "/data/input_output_data_tc1.pickle", "rb") as data:
        dataset = pickle.load(data)
    return dataset[0], dataset[1], dataset[2], dataset[3]


## generate random samples within the input polygon
def gen_rand_points_within_poly(
    poly, num_points, unif2edge=0.75, edge_scale=0.7
):
    # pylint: disable=invalid-name
    """_summary_

    Args:
        poly (_type_): _description_
        num_points (_type_): _description_
        unif2edge (float, optional): _description_. Defaults to 0.75.
        edge_scale (float, optional): _description_. Defaults to 0.7.

    Returns:
        _type_: _description_
    """
    min_x, min_y, max_x, max_y = poly.bounds
    poly_a = affinity.scale(poly, xfact=edge_scale, yfact=edge_scale)

    num_pts_unif = int(unif2edge * num_points)

    x = np.ones((num_points, 3))  # (x,y,b)

    # uniformly distributed points
    count = 0
    while count < num_pts_unif:
        rand_pt = [
            np.random.uniform(min_x, max_x),
            np.random.uniform(min_y, max_y),
        ]
        random_point = Point(rand_pt)
        if random_point.within(poly):
            x[count, 0:2] = np.array(rand_pt).flatten()
            count += 1

    # edge points
    while count < num_points:
        while True:
            rand_pt = [
                np.random.uniform(min_x, max_x),
                np.random.uniform(min_y, max_y),
            ]
            random_point = Point(rand_pt)
            if random_point.within(poly) and not random_point.within(poly_a):
                break
        x[count, 0:2] = np.array(rand_pt).flatten()
        count += 1

    np.random.shuffle(x)
    return x


class Batch:
    # pylint: disable=too-few-public-methods
    """_summary_"""

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
        # indices = np.random.choice(range(self.size), self.batch_size)  # sampling with replacement
        return (
            self.X[train_dataset, :],
            self.Y[train_dataset, :],
            self.X[test_dataset, :],
            self.Y[test_dataset, :],
        )


def label_output_inside(
    poly_const, inp_data, out_data, bound_error=0.3, mode="finetune"
):
    # pylint: disable=invalid-name
    """Project the target points that lie outside of the contrained poly
    to the poly's surface.
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
    ex_points = np.array(poly_const.exterior.coords)
    P = np.diag(np.ones(2))
    # get A and b matrices
    hull = ConvexHull(ex_points)
    A = np.array(-hull.equations[:, :-1], dtype=float)
    b = np.array(hull.equations[:, -1], dtype=float)

    for i in range(out_data.shape[0]):
        if not Point([out_data[i, 0], out_data[i, 1]]).within(poly_const):
            sol, _, _, _, _, _ = solve_qp(
                P, out_data[i, 0:2], A.T, b, meq=0, factorized=True
            )
            sol = (
                (
                    (poly_const.exterior.distance(Point([sol[0], sol[1]])))
                    + bound_error
                )
                / np.linalg.norm(sol - out_data[i, 0:2])
            ) * (sol - out_data[i, 0:2]) + sol
            # if not Point([sol[0], sol[1]]).within(
            #     poly_const
            # ):  # captures the within error
            #     sol = (
            #         ((poly_const.exterior.distance(Point([sol[0], sol[1]]))) + 0.00001)
            #         / np.linalg.norm(sol - out_data[i, 0:2])
            #     ) * (sol - out_data[i, 0:2]) + sol

            out_data_new.append(np.append(sol, [1]))
            inp_data_new.append(inp_data[i, :])
        else:
            if mode == "retrain":
                out_data_new.append(out_data[i, :])
                inp_data_new.append(inp_data[i, :])

    return np.array(inp_data_new), np.array(out_data_new)


def model_eval(model_new, model_orig, path_read, poly_const):
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
            path_read + "/data/input_output_data_inside_train_tc1.pickle"
        )
        or os.path.exists(
            path_read + "/data/input_output_data_outside_train_tc1.pickle"
        )
        or os.path.exists(
            path_read + "/data/input_output_data_inside_test_tc1.pickle"
        )
        or os.path.exists(
            path_read + "/data/input_output_data_outside_test_tc1.pickle"
        )
    ):
        raise ImportError(
            "inside-outside datasets for test and train should be generated first in path {path_read}/data/"
        )
    # with open(
    #     path_read + "/data/input_output_data_inside_train_tc1.pickle", "rb"
    # ) as data:
    #     train_inside = pickle.load(data)
    with open(
        path_read + "/data/input_output_data_outside_train_tc1.pickle", "rb"
    ) as data:
        train_out = pickle.load(data)
    with open(
        path_read + "/data/input_output_data_inside_test_tc1.pickle", "rb"
    ) as data:
        test_in = pickle.load(data)
    with open(
        path_read + "/data/input_output_data_outside_test_tc1.pickle", "rb"
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
        if not Point(y_train_out_pred[i][0], y_train_out_pred[i][1]).within(
            poly_const
        ):
            if (
                not poly_const.exterior.distance(
                    Point(y_train_out_pred[i][0], y_train_out_pred[i][1])
                )
                < 0.00001
            ):
                num_corrected_train -= 1

    acc_buggy_train = num_corrected_train / num_buggy_train

    # new model accuracy on buggy data - test
    y_test_out_pred = model_new.predict(test_out[0])
    num_buggy_test = test_out[1].shape[0]
    num_corrected_test = num_buggy_test

    for i in range(num_buggy_test):
        if not Point(y_test_out_pred[i][0], y_test_out_pred[i][1]).within(
            poly_const
        ):
            if (
                not poly_const.exterior.distance(
                    Point(y_test_out_pred[i][0], y_test_out_pred[i][1])
                )
                < 0.00001
            ):
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


def plot_dataset(polys, out_dataset, label="training"):
    """_summary_

    Args:
        polys (_type_): _description_
        out_dataset (_type_): _description_
        label (str, optional): _description_. Defaults to "training".
    """
    print("----------------------")
    print(f"Data samples Visualization ({label})")
    plt.rcParams["text.usetex"] = False
    mpl.style.use("seaborn")

    poly_labels = ["Original Set", "Target Set", "Constraint Set"]
    colors = ["plum", "tab:blue", "tab:red"]
    # plot polys
    for i, poly in enumerate(polys):
        x_poly_bound, y_poly_bound = poly.exterior.xy
        plt.plot(
            x_poly_bound,
            y_poly_bound,
            color=colors[i],
            alpha=0.7,
            linewidth=3,
            solid_capstyle="round",
            zorder=2,
            label=poly_labels[i],
        )
    plot_labels = ["Original Target", "Predicted Target"]
    for i, data in enumerate(out_dataset):
        plt.scatter(
            data[:, 0],
            data[:, 1],
            color=colors[i],
            label=plot_labels[i],
        )

    plt.legend(loc="upper left", frameon=False, fontsize=20)
    plt.title(f"In-place Rotation ({label} dataset)", fontsize=25)
    plt.xlabel("x", fontsize=25)
    plt.ylabel("y", fontsize=25)
    plt.show()


def transform_mesh(x, y):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    # transform matrices
    translate1 = np.array([[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]])
    translate2 = np.array([[1, 0, -2.5], [0, 1, -2.5], [0, 0, 1]])
    rotate = np.array(
        [
            [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
            [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
            [0, 0, 1],
        ]
    )
    row, col = x.shape
    x_out = np.zeros((row, col))
    y_out = np.zeros((row, col))
    for r in range(row):
        for c in range(col):
            temp = np.matmul(
                np.matmul(np.matmul(translate1, rotate), translate2),
                np.array([[x[r, c], y[r, c], 1]]).T,
            )

            x_out[r, c] = temp.flatten()[0]
            y_out[r, c] = temp.flatten()[1]
    return x_out, y_out


def plot_meshgird(
    poly, mesh_data, mesh_color, title, path_write, mesh_orig=None
):
    """_summary_

    Args:
        poly (_type_): _description_
        mesh_data (_type_): _description_
        mesh_color (_type_): _description_
        title (_type_): _description_
    """
    x_poly, y_poly = poly.exterior.xy
    plt.title(title)
    plt.plot(
        x_poly,
        y_poly,
        color="black",
        alpha=0.7,
        linewidth=1,
        solid_capstyle="round",
        zorder=2,
    )
    for r in range(mesh_data[0].shape[0]):
        for c in range(mesh_data[0].shape[1]):
            # if mesh_orig:
            if mesh_orig:
                plt.scatter(
                    mesh_orig[0][r, c],
                    mesh_orig[1][r, c],
                    color=(0.6, 0.6, 1),
                    linewidth=3,
                )
                dist = np.linalg.norm(
                    np.array([mesh_orig[0][r, c], mesh_orig[1][r, c]])
                    - np.array([mesh_data[0][r, c], mesh_data[1][r, c]])
                )
                RGB = colorsys.hsv_to_rgb(0.0, dist + 0.1, 1.0)
                plt.scatter(
                    mesh_data[0][r, c],
                    mesh_data[1][r, c],
                    color=RGB,
                    linewidth=3,
                )
            else:
                RGB = colorsys.hsv_to_rgb(0.0, 1.0, 1.0)
                plt.scatter(
                    mesh_data[0][r, c],
                    mesh_data[1][r, c],
                    color=RGB,
                    linewidth=3,
                )
    plt.axis("off")
    plt.savefig(path_write + f"/mesh_{title}.jpg")
    plt.close()


def give_mesh2direc(mesh_data, mesh_orig):
    """_summary_

    Args:
        mesh_data (_type_): _description_
        mesh_orig (_type_): _description_

    Returns:
        _type_: _description_
    """

    row, col = mesh_data[0].shape
    x_direc = np.zeros((row, col))
    y_direc = np.zeros((row, col))

    for r in range(mesh_data[0].shape[0]):
        for c in range(mesh_data[0].shape[1]):
            temp = np.array(
                [mesh_data[0][r, c], mesh_data[1][r, c]]
            ) - np.array([mesh_orig[0][r, c], mesh_orig[1][r, c]])
            x_direc[r, c] = temp[0]
            y_direc[r, c] = temp[1]

    return x_direc, y_direc


def give_mesh2dist(mesh_data, mesh_orig):
    """_summary_

    Args:
        mesh_data (_type_): _description_
        mesh_orig (_type_): _description_

    Returns:
        _type_: _description_
    """

    row, col = mesh_data[0].shape
    dist_mesh = np.zeros((row, col))

    for r in range(mesh_data[0].shape[0]):
        for c in range(mesh_data[0].shape[1]):
            temp = np.array(
                [mesh_data[0][r, c], mesh_data[1][r, c]]
            ) - np.array([mesh_orig[0][r, c], mesh_orig[1][r, c]])
            dist_mesh[r, c] = np.linalg.norm(temp)

    return dist_mesh


def give_mix_samples(discretization=10):
    """_summary_

    Args:
        discretization (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    x_train = []
    y_train = []
    # transform matrices
    translate1 = np.array([[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]])
    translate2 = np.array([[1, 0, -2.5], [0, 1, -2.5], [0, 0, 1]])
    rotate = np.array(
        [
            [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
            [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
            [0, 0, 1],
        ]
    )

    x1_mesh, x2_mesh = np.meshgrid(
        np.linspace(1, 4, discretization), np.linspace(1, 4, discretization)
    )
    row, col = x1_mesh.shape
    for r in range(row):
        for c in range(col):
            temp = np.matmul(
                np.matmul(np.matmul(translate1, rotate), translate2),
                np.array([[x1_mesh[r, c], x2_mesh[r, c], 1]]).T,
            )
            x_train.append(np.array([x1_mesh[r, c], x2_mesh[r, c], 1]))
            y_train.append(np.array([temp.flatten()[0], temp.flatten()[1], 1]))

    poly_orig, _, _ = give_polys()
    x_rand = gen_rand_points_within_poly(
        poly_orig, 100, unif2edge=0.01, edge_scale=0.7
    )
    y_rand = np.matmul(
        np.matmul(np.matmul(translate1, rotate), translate2),
        x_rand.T,
    ).T
    return np.vstack((x_train, x_rand)), np.vstack((y_train, y_rand))


def give_equidistance_samples(discretization=15):
    """_summary_

    Args:
        discretization (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    x_train = []
    y_train = []
    # transform matrices
    translate1 = np.array([[1, 0, 2.5], [0, 1, 2.5], [0, 0, 1]])
    translate2 = np.array([[1, 0, -2.5], [0, 1, -2.5], [0, 0, 1]])
    rotate = np.array(
        [
            [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
            [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
            [0, 0, 1],
        ]
    )

    x1_mesh, x2_mesh = np.meshgrid(
        np.linspace(1, 4, discretization), np.linspace(1, 4, discretization)
    )
    row, col = x1_mesh.shape
    for r in range(row):
        for c in range(col):
            temp = np.matmul(
                np.matmul(np.matmul(translate1, rotate), translate2),
                np.array([[x1_mesh[r, c], x2_mesh[r, c], 1]]).T,
            )
            x_train.append(np.array([x1_mesh[r, c], x2_mesh[r, c], 1]))
            y_train.append(np.array([temp.flatten()[0], temp.flatten()[1], 1]))

    return np.array(x_train), np.array(y_train)


def plot_quiver(
    poly_orig, poly_const, mesh_data, title, path_write, mesh_orig
):
    """_summary_

    Args:
        poly (_type_): _description_
        mesh_data (_type_): _description_
        mesh_color (_type_): _description_
        title (_type_): _description_
    """
    cdict = {
        "red": ((0.0, 0.25, 0.25), (0.02, 0.59, 0.59), (1.0, 1.0, 1.0)),
        "green": ((0.0, 0.0, 0.0), (0.02, 0.45, 0.45), (1.0, 0.97, 0.97)),
        "blue": ((0.0, 1.0, 1.0), (0.02, 0.75, 0.75), (1.0, 0.45, 0.45)),
    }

    fig, ax = plt.subplots(figsize=(7, 7))
    c = ax.pcolormesh(
        mesh_orig[0],
        mesh_orig[1],
        give_mesh2dist(mesh_data, mesh_orig),
        cmap=mpl.colors.LinearSegmentedColormap("my_colormap", cdict, 1024),
        vmin=0,
        vmax=0.3,
        shading="gouraud",
    )
    fig.colorbar(c)
    x_poly, y_poly = poly_orig.exterior.xy
    plt.title(title)
    ax.plot(
        x_poly,
        y_poly,
        color="black",
        alpha=0.7,
        linewidth=2,
        solid_capstyle="round",
        zorder=2,
        label="original set",
    )
    x_poly, y_poly = poly_const.exterior.xy
    ax.plot(
        x_poly,
        y_poly,
        color="blue",
        alpha=0.7,
        linewidth=2,
        solid_capstyle="round",
        zorder=2,
        label="constrained set",
    )

    u, v = give_mesh2direc(mesh_data, mesh_orig)
    dist_mat = give_mesh2dist(mesh_data, mesh_orig)
    ax.quiver(
        mesh_orig[0],
        mesh_orig[1],
        u,
        v,
        angles="xy",
        linewidth=1,
        scale_units="xy",
        scale=1,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend()
    plt.savefig(path_write + f"/quiver_{title}.eps")
    plt.close()


def net_meshgrid_prediction(x, y, model):
    """_summary_

    Args:
        x (_type_): _description_
        y (_type_): _description_
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    row, col = x.shape
    x_out = np.zeros((row, col))
    y_out = np.zeros((row, col))
    for r in range(row):
        for c in range(col):
            temp = model.predict(np.array([[x[r, c], y[r, c], 1]]))
            x_out[r, c] = temp.flatten()[0]
            y_out[r, c] = temp.flatten()[1]
    return x_out, y_out
