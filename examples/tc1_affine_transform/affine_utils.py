from shapely.geometry import Polygon, Point
import numpy as np
import shapely.affinity as affinity
from sklearn.model_selection import train_test_split

## generate random samples within the input polygon
def gen_rand_points_within_poly(poly, num_points, unif2edge = 0.75, edge_scale = 0.7):
    min_x, min_y, max_x, max_y = poly.bounds
    poly_a = affinity.scale(poly, xfact = edge_scale, yfact = edge_scale)

    num_pts_unif = int(unif2edge * num_points)

    x = np.ones((num_points, 3))     #(x,y,b)

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
            if random_point.within(poly) and not random_point.within(poly_a): break
        x[count, 0:2] = np.array(rand_pt).flatten()
        count += 1
    
    np.random.shuffle(x)
    return x

class Batch(object):
    def __init__(self, X_col, Y_col, batch_size_val):
        self.X = X_col
        self.Y = Y_col
        self.size = X_col.shape[0]
        self.train_size = batch_size_val
        self.test_size = self.size - batch_size_val

    def getBatch(self):
        values = range(self.size)
        train_dataset, test_dataset = train_test_split(values, train_size=self.train_size, test_size=self.test_size)
        # indices = np.random.choice(range(self.size), self.batch_size)  # sampling with replacement
        return self.X[train_dataset, :], self.Y[train_dataset, :], self.X[test_dataset, :], self.Y[test_dataset, :]