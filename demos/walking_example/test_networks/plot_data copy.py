import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import interp1d
import pickle
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as tick

plt.rcParams.update({"text.usetex": True})


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    return data


def give_dist(x_train, x_test):
    dist = []
    for i in range(x_test.shape[0]):
        dist.append(
            np.min(np.linalg.norm(x_train[:, :-1] - x_test[i, :-1], axis=1))
        )

    dist = np.array(dist)
    return dist


# load original dataset global

with open(
    os.path.dirname(os.path.realpath(__file__))
    + f"/dynamic_constraint/data/repair_dataset_6_11_2022_10_56_16.pickle",
    "rb",
) as data:
    dataset = pickle.load(data)
x_rep_dyn = dataset[0]
y_rep_dyn = dataset[1]

with open(
    os.path.dirname(os.path.realpath(__file__))
    + f"/global_constraint/data/repair_dataset_4_6_8_2022_18_41_59.pickle",
    "rb",
) as data:
    dataset = pickle.load(data)
x_rep_glob = dataset[0]
y_rep_glob = dataset[1]


# GET PHASE FROM DATA
# n=20364

# with open('demos/walking_example/data/heel_strikes.csv') as csv_file:
#     data = np.asarray(list(csv.reader(csv_file, delimiter=',')), dtype=np.float32)

# phase=np.array([]).reshape(0,)
# for i in range(data.shape[0]-1):
#     temp = np.linspace(0,1,int(data[i+1]-data[i])+1
#     phase=np.concatenate((phase,temp[:-1]))
#     pass
# np.savetxt("demos/walking_example/data/phase.csv", phase, delimiter=",")


# SET LIMITS
global_data = loadData(
    "demos/walking_example/test_networks/global_constraint_test.csv"
)
global_data = global_data[8034:8734, :]
dyanmic_data = loadData(
    "demos/walking_example/test_networks/dynamic_constraint_test.csv"
)[8000:8700, :]
data_size = global_data.shape[0]

Dfem = dyanmic_data[:, 0:2]
Dtib = dyanmic_data[:, 2:4]
Dankle = dyanmic_data[:, 4]
Dfem = (Dfem - Dfem.mean(0)) / Dfem.std(0)
Dtib = (Dtib - Dtib.mean(0)) / Dtib.std(0)
observations = np.concatenate((Dfem, Dtib), axis=1)
observations = np.concatenate(
    (
        observations,
        Dankle.reshape(observations.shape[0], 1),
    ),
    axis=1,
)
n_train = observations.shape[0] - 1
window_size = 10
train_observation = np.array([]).reshape(0, 5 * window_size)
for i in range(Dfem.shape[0] - window_size):
    temp_obs = np.array([]).reshape(1, 0)
    for j in range(window_size):
        temp_obs = np.concatenate(
            (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
        )
    train_observation = np.concatenate((train_observation, temp_obs), axis=0)


angle_global = global_data[:, -1].flatten()[window_size:] - 1
angle_dynamic = dyanmic_data[:, -1].flatten()[window_size:]
error_global = 0.7 * (angle_global[1:] - angle_global[:-1])
error_dynamic = 0.7 * (angle_dynamic[1:] - angle_dynamic[:-1])
# angle_global = (angle_global - angle_global.mean(0)) / angle_global.std(0)
# angle_dynamic = (angle_dynamic - angle_dynamic.mean(0)) / angle_dynamic.std(0)
error_dynamic = error_dynamic
angle_dynamic = angle_dynamic

distance = give_dist(x_rep_dyn, train_observation)
distance = distance / distance.max()
# plt.plot(angle_global)
# plt.plot(angle_dynamic)
# # plt.plot(0.7 * angle_error)
# plt.axhline(
#     y=2,
#     color="black",
#     linewidth=1,
#     linestyle=(0, (5, 7)),
#     alpha=1,
# )  # upper bound
# plt.axhline(
#     y=-2,
#     color="black",
#     linewidth=1,
#     linestyle=(0, (5, 7)),
#     alpha=1,
# )  # upper bound
# plt.show()
T = 0.07
x = np.linspace(0, T * (data_size - window_size), data_size - window_size)
y = np.linspace(-20, 30, 100)

z = [distance for j in y]

# CS = plt.contourf(
#     x, y, z, 10, cmap=plt.get_cmap("Greys")
# )  # \[-1, -0.1, 0, 0.1\],
# plt.colorbar(CS)


fig = plt.figure(figsize=(8, 5))
# color_bar = "Greens"
color_bar = mpl.cm.Greys(np.linspace(0, 0.6, 100))
color_bar = mpl.colors.ListedColormap(color_bar[10:, :-1])

color_orig = "#DC143C"
color_lay3 = "k"
color_lay4 = "#800080"
color_test = "black"
color_xline = "k"
color_fill = "#D4D4D4"
colorbar_range = 100
line_width = 1.5
font_size = 12
x_min = 200
x_max = data_size - window_size
gs = fig.add_gridspec(2, 1)
ax00 = fig.add_subplot(gs[0, 0])
ax10 = fig.add_subplot(gs[1, 0])
# ax02 = fig.add_subplot(gs[0, 2])
# ax12 = fig.add_subplot(gs[1, 2])

ax00.get_shared_x_axes().join(ax00, ax10)
ax10.get_shared_x_axes().join(ax00, ax10)
# ax02.get_shared_x_axes().join(ax02, ax12)
# ax12.get_shared_x_axes().join(ax02, ax12)
len = angle_global.flatten().shape[0] - x_min
time = np.linspace(0, len * T, len, endpoint=False)
# plot bound 2 plots
ax00.plot(
    time,
    angle_global.flatten()[x_min:],
    label="Rep. global constraint",
    color=color_orig,
    linewidth=1.5,
    # linestyle="dashed",
)
ax00.plot(
    time,
    angle_dynamic.flatten()[x_min:],
    label="Rep. input-output constraint",
    color=color_lay3,
    linewidth=1.5,
)
CS = ax00.contourf(
    x, y, z, colorbar_range, cmap=color_bar, vmin=0.0, vmax=1.0
)  # \[-1, -0.1, 0, 0.1\],
# ax00.colorbar(CS)
ax00.axhline(
    y=24.0, color=color_xline, linewidth=1.5, linestyle="dashed"
)  # upper bound
ax00.axhline(
    y=-14.0, color=color_xline, linewidth=1.5, linestyle="dashed"
)  # lower bound

# ax00.fill_between(
#     np.linspace(
#         0, error_global.shape[0], error_global.shape[0], endpoint=True
#     ),
#     0,
#     1,
#     where=np.abs(error_global.flatten()) > 2.0,
#     color=color_fill,
#     alpha=1,
#     transform=ax00.get_xaxis_transform(),
#     label="Violated region",
# )
ax00.set_ylabel("Control [deg]", fontsize=font_size)
ax00.grid(alpha=0.8, linestyle="dashed")
ax00.set_ylim([-17.0, 27.0])
ax00.set_yticks(np.linspace(-15.0, 25.0, 3, endpoint=True))
ax00.set_xticks(np.linspace(0, T * (x_max - x_min), 5, endpoint=True))
ax00.tick_params(axis="y", which="major", labelsize=font_size)

ax00.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax00.xaxis.set_ticklabels([])
ax00.yaxis.set_major_formatter(FormatStrFormatter("%d"))
# ax00.set_title("Control bound = 2", fontsize=font_size)

ax10.plot(
    time[:-1],
    error_global[x_min:],
    color=color_orig,
    linewidth=1.5,
)
# ax10.plot(
#     delta_u_laye4_bound2,
#     color="blue",
#     linewidth=1.5,
# )
ax10.plot(
    time[:-1],
    error_dynamic[x_min:],
    color=color_lay3,
    linewidth=1.5,
)
ax10.contourf(x, y, z, colorbar_range, cmap=color_bar, vmin=0.0, vmax=1.0)
# ax10.fill_between(
#     np.linspace(
#         0, error_global.shape[0], error_global.shape[0], endpoint=True
#     ),
#     0,
#     1,
#     where=np.abs(error_global.flatten()) > 2.0,
#     color=color_fill,
#     alpha=0.8,
#     transform=ax10.get_xaxis_transform(),
# )
ax10.axhline(
    y=2.0, color=color_xline, linewidth=1.5, linestyle="dashed"
)  # upper bound
ax10.axhline(
    y=-2.0, color=color_xline, linewidth=1.5, linestyle="dashed"
)  # lower bound
ax10.set_ylabel("Control rate [deg/s]", fontsize=font_size)
ax10.grid(alpha=0.8, linestyle="dashed")
ax10.set_xlabel("Time (s)", fontsize=font_size)
ax10.set_xlim([0, T * (x_max - x_min)])
ax10.set_ylim([-3, 3])
ax10.set_xticks(np.linspace(0, T * (x_max - x_min), 5, endpoint=True))
ax10.set_yticks(np.linspace(-2, 2, 3, endpoint=True))
ax10.tick_params(axis="x", labelsize=font_size)
ax10.tick_params(axis="y", labelsize=font_size)
ax10.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax10.yaxis.set_major_formatter(FormatStrFormatter("%d"))
v = np.linspace(0, 1, 5, endpoint=True)
cbar = fig.colorbar(CS, ax=[ax00, ax10], alpha=0.5, ticks=v)
cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter("%.1f"))

lines, labels = ax00.get_legend_handles_labels()
leg = fig.legend(
    lines,
    labels,
    loc="center",
    # bbox_to_anchor=(0.5, -0.5),
    bbox_to_anchor=(0.5, 0.0),
    bbox_transform=fig.transFigure,
    ncol=2,
    fontsize=font_size,
)
leg.get_frame().set_facecolor("white")

plt.show()
