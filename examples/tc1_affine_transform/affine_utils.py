import numpy as np
import shapely.affinity as affinity
from sklearn.model_selection import train_test_split
from quadprog import solve_qp
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from matplotlib import pyplot as plt
import matplotlib as mpl

## generate random samples within the input polygon
def gen_rand_points_within_poly(poly, num_points, unif2edge=0.75, edge_scale=0.7):
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
        rand_pt = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
        random_point = Point(rand_pt)
        if random_point.within(poly):
            x[count, 0:2] = np.array(rand_pt).flatten()
            count += 1

    # edge points
    while count < num_points:
        while True:
            rand_pt = [np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y)]
            random_point = Point(rand_pt)
            if random_point.within(poly) and not random_point.within(poly_a):
                break
        x[count, 0:2] = np.array(rand_pt).flatten()
        count += 1

    np.random.shuffle(x)
    return x


class Batch:
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


def label_output_inside(poly_const, inp_data, out_data, mode="finetune"):
    """Project the target points that lie outside of the contrained poly
    to the poly's surface.
    Outputs either all target data points(retrain) or just projected points (finetune).

    Args:
        poly_const (_type_): _description_
        out_data (_type_): _description_
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
                ((poly_const.exterior.distance(Point([sol[0], sol[1]]))) + 0.15)
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


def plot_history(his, include_validaation=False):
    print("----------------------")
    print("History Visualization")
    plt.rcParams["text.usetex"] = True
    mpl.style.use("seaborn")

    ## loss plotting
    results_train_loss = his.history["loss"]
    plt.plot(results_train_loss, color="red", label="training loss")
    if include_validaation:
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
    if include_validaation:
        results_valid_acc = his.history["val_accuracy"]
        plt.plot(results_valid_acc, color="blue", label="validation accuracy")
    plt.title("Accuracy Function Output (fine-tuning the last layer)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(loc="upper left", frameon=False)
    plt.show()


def plot_dataset(polys, out_dataset, label="training"):
    print("----------------------")
    print(f"Data samples Visualization ({label})")
    plt.rcParams["text.usetex"] = True
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
