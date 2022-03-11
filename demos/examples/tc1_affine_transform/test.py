import os
from nnreplayer.utils.utils import constraints_class
from tensorflow import keras
from shapely.affinity import scale
from nnreplayer.repair.repair_weights_class import NNRepair
from affine_utils import original_data_loader, give_polys, give_constraints
from nnreplayer.utils.options import Options
from matplotlib import pyplot as plt


path_read = os.path.dirname(os.path.realpath(__file__)) + "/tc1"
model_orig = keras.models.load_model(path_read + "/original_net/model_2")
repair_obj = NNRepair(model_orig)
x_repair, y_repair, _, _ = original_data_loader()
poly_orig, poly_trans, poly_const = give_polys()
A, b = give_constraints(scale(poly_const, xfact=1, yfact=1, origin="center"))
constraint_inside = constraints_class("inside", A, b)
output_constraint_list = [constraint_inside]
repair_obj.compile(
    x_repair,
    y_repair,
    3,
    output_constraint_list=output_constraint_list,
    max_weight_bound=5,
)
options = Options(
    "gdp.bigm",
    "gurobi",
    "python",
    "keras",
    {
        "timelimit": 500,
        "mipgap": 0.001,
        "mipfocus": 2,
        "improvestarttime": 3300,
        # "logfile": f"/opt_log_layer{3}.log",
    },
)
# display model
out_model = repair_obj.repair(options)
repair_obj.print_opt_model(path_read + "/repair_net/logs")
y_new = out_model.predict(x_repair)
x_poly_bound, y_poly_bound = poly_const.exterior.xy
plt.plot(x_poly_bound, y_poly_bound)
plt.scatter(y_new[:, 0], y_new[:, 1])
plt.scatter(y_repair[:, 0], y_repair[:, 1])
plt.show()
