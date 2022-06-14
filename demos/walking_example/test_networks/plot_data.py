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
error_global = 0.7 * (angle_global[1:] - angle_global[:-1])
error_dynamic = 0.7 * (angle_dynamic[1:] - angle_dynamic[:-1])
# angle_global = (angle_global - angle_global.mean(0)) / angle_global.std(0)
# angle_dynamic = (angle_dynamic - angle_dynamic.mean(0)) / angle_dynamic.std(0)
error_dynamic = error_dynamic
angle_dynamic = angle_dynamic
plt.plot(angle_global)
plt.plot(angle_dynamic)
# plt.plot(0.7 * angle_error)
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


fig = plt.figure(figsize=(13, 5))
color_orig = "#2E8B57"
color_lay3 = "#DC143C"
color_lay4 = "#800080"
color_test = "black"
color_xline = "#696969"
color_fill = "#D4D4D4"
line_width = 2

gs = fig.add_gridspec(2, 1)
ax00 = fig.add_subplot(gs[0, 0])
ax10 = fig.add_subplot(gs[1, 0])
# ax02 = fig.add_subplot(gs[0, 2])
# ax12 = fig.add_subplot(gs[1, 2])

ax00.get_shared_x_axes().join(ax00, ax10)
ax10.get_shared_x_axes().join(ax00, ax10)
# ax02.get_shared_x_axes().join(ax02, ax12)
# ax12.get_shared_x_axes().join(ax02, ax12)

# plot bound 2 plots
ax00.plot(
    angle_global.flatten(),
    label="Global",
    color=color_orig,
    linewidth=1.5,
    linestyle="dashed",
)
ax00.plot(
    angle_dynamic.flatten(),
    label="Input-output",
    color=color_lay3,
    linewidth=1.5,
)
ax00.axhline(
    y=24.0, color="#8B8878", linewidth=1.5, linestyle="dashed"
)  # upper bound
ax00.axhline(
    y=-14.0, color="#8B8878", linewidth=1.5, linestyle="dashed"
)  # lower bound

ax00.fill_between(
    np.linspace(
        0, error_global.shape[0], error_global.shape[0], endpoint=True
    ),
    0,
    1,
    where=np.abs(error_global.flatten()) > 2.0,
    color=color_fill,
    alpha=0.8,
    transform=ax00.get_xaxis_transform(),
    label="Violated region",
)
ax00.set_ylabel("Control (rad)", fontsize=16)
ax00.grid(alpha=0.8, linestyle="dashed")
ax00.set_ylim([-15.0, 25.0])
ax00.set_yticks(np.linspace(-15.0, 25.0, 5, endpoint=True))
ax00.xaxis.set_ticklabels([])
ax00.tick_params(axis="both", which="major", labelsize=16)
ax00.set_title("Control bound = 2", fontsize=16)

ax10.plot(
    error_global,
    color=color_orig,
    linewidth=1.5,
)
# ax10.plot(
#     delta_u_laye4_bound2,
#     color="blue",
#     linewidth=1.5,
# )
ax10.plot(
    error_dynamic,
    color=color_lay3,
    linewidth=1.5,
)
ax10.fill_between(
    np.linspace(
        0, error_global.shape[0], error_global.shape[0], endpoint=True
    ),
    0,
    1,
    where=np.abs(error_global.flatten()) > 2.0,
    color=color_fill,
    alpha=0.8,
    transform=ax10.get_xaxis_transform(),
)
ax10.axhline(
    y=2.0, color="#8B8878", linewidth=1.5, linestyle="dashed"
)  # upper bound
ax10.axhline(
    y=-2.0, color="#8B8878", linewidth=1.5, linestyle="dashed"
)  # lower bound
ax10.set_ylabel("Control rate (rad/s)", fontsize=16)
ax10.grid(alpha=0.8, linestyle="dashed")
ax10.set_xlabel("Time (s)", fontsize=16)
ax10.set_xlim([11740, 11900])
ax10.set_ylim([-3, 3])
ax10.set_xticks(np.linspace(11740, 11840, 5, endpoint=True))
ax10.set_yticks(np.linspace(-2, 2, 5, endpoint=True))
ax10.tick_params(axis="x", labelsize=16)
ax10.tick_params(axis="y", labelsize=16)

plt.show()
