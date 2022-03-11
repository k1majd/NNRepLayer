import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
poly3 = Polygon([(.45, .1), (.45, .25), (.55, .25), (.55, .1)])

# get the coordinates of the exterior points of the polytope
ex_points = np.array(poly3.exterior.coords)

# get A and b matrices: A*x <= b
hull = ConvexHull(ex_points)
eqs = np.array(hull.equations)
A = eqs[0:eqs.shape[0],0:eqs.shape[1]-1]  
b = -eqs[0:eqs.shape[0],-1]
b = np.array([b]).T


def generate_constraints(A,b):

    assert len(b.shape) == 2, 'Shape of b must be N x 1 dimension'
    A_m, A_n = A.shape
    b_n, b_p = b.shape
    assert A_m == b_n, "First dimension of A (= {}) should be equal to be first dimesnion of b(= {}).".format(A_m, b_n)
    assert b_p == 1, "Second dimension of b (={}) should be 1".format(b_p)

    def_string = ""
    add_attr_list = []
    for i in range(A_m):
        single_def_string = ""
        temp_string = []
        
        for j in range(A_n):
            t_st = "A[{},{}]*getattr(model, x4)[i, {}]".format(i,j,j,i)
            temp_string.append(t_st)
        add_attr_string = "setattr(self.model, 'keep_inside_constraint{}'+str(l),pyo.Constraint(range(m), rule=constraint_inside{}))".format(i,i)
        add_attr_list.append(add_attr_string)
        return_string = "(" + " + ".join(temp_string) + "  - b[{}] <= 0)".format(i)
        single_def_string = "def constraint_inside{}(model, i):\n\treturn ".format(i) + return_string 
        def_string = def_string + single_def_string + "\n\n"
    
    attr_string = "\n".join(add_attr_list)
    
    def_string += attr_string

    return def_string

fin_string = generate_constraints(A,b)
print(fin_string)