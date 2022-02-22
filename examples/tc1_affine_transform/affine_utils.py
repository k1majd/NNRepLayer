import numpy as np
import shapely.affinity as affinity
from sklearn.model_selection import train_test_split
import cvxopt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point

## generate random samples within the input polygon
def gen_rand_points_within_poly(poly, num_points, unif2edge=0.75, edge_scale=0.7):
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
    def __init__(self, X_col, Y_col, batch_size_val):
        self.X = X_col
        self.Y = Y_col
        self.size = X_col.shape[0]
        self.train_size = batch_size_val
        self.test_size = self.size - batch_size_val

    def getBatch(self):
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


def label_output_inside(poly_const, out_data):
    # get the coordinates of the exterior points of the polytope
    ex_points = np.array(poly_const.exterior.coords)
    P = np.diag(np.ones(2))
    # get A and b matrices
    hull = ConvexHull(ex_points)
    eqs = np.array(hull.equations)
    A = eqs[0 : eqs.shape[0], 0 : eqs.shape[1] - 1]
    b = -eqs[0 : eqs.shape[0], -1]

    for i in range(out_data.shape[0]):
        if not Point([out_data[i, 0], out_data[i, 1]]).within(poly_const):
            q = out_data[i, 0:2]
            sol = solve_qp(P, q, A=A, b=b)


def solve_qp(P, q, G=None, h=None, A=None, b=None):
    # pylint: disable=invalid-name
    # pylint: disable=too-many-arguments

    """_summary_

    Args:
        P (_type_): _description_
        q (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        h (_type_, optional): _description_. Defaults to None.
        A (_type_, optional): _description_. Defaults to None.
        b (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    P = 0.5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options["show_progress"] = False
    cvxopt.solvers.options["maxiters"] = 100
    sol = cvxopt.solvers.qp(*args)
    if "optimal" not in sol["status"]:
        return None
    return np.array(sol["x"]).reshape((P.shape[1],))
