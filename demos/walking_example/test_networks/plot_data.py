import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import interp1d


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    return data


# GET PHASE FROM DATA
# n=20364

# with open('demos/walking_example/data/heel_strikes.csv') as csv_file:
#     data = np.asarray(list(csv.reader(csv_file, delimiter=',')), dtype=np.float32)

# phase=np.array([]).reshape(0,)
# for i in range(data.shape[0]-1):
#     temp = np.linspace(0,1,int(data[i+1]-data[i])+1)
#     phase=np.concatenate((phase,temp[:-1]))
#     pass
# np.savetxt("demos/walking_example/data/phase.csv", phase, delimiter=",")


# SET LIMITS
global_data = loadData(
    "demos/walking_example/test_networks/global_constraint_test.csv"
)
dyanmic_data = loadData(
    "demos/walking_example/test_networks/dynamic_constraint_test.csv"
)

angle_global = global_data[:, -1].flatten()
angle_dynamic = dyanmic_data[:, -1].flatten()
angle_error = angle_dynamic[1:] - angle_dynamic[:-1]
plt.plot(0.667 * angle_error)
plt.axhline(
    y=2,
    color="black",
    linewidth=1,
    linestyle=(0, (5, 7)),
    alpha=1,
)  # upper bound
plt.axhline(
    y=-2,
    color="black",
    linewidth=1,
    linestyle=(0, (5, 7)),
    alpha=1,
)  # upper bound
plt.show()

# np.savetxt("demos/walking_example/data/GeoffFTF_limits.csv", limits, delimiter=",")
