import numpy as np
import os
import csv
import matplotlib.pyplot as plt


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    return data


def generateDataWindow(window_size=10):
    Dfem = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/GeoffFTF_1.csv"
    )
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/GeoffFTF_2.csv"
    )
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/GeoffFTF_3.csv"
    )
    n = 20364
    Dankle = np.subtract(Dtib[:n, 1], Dfut[:n, 1])
    observations = np.concatenate((Dfem[:n, 1:], Dtib[:n, 1:]), axis=1)
    observations = (observations - observations.mean(0)) / observations.std(0)
    controls = Dankle  # (Dankle - Dankle.mean(0))/Dankle.std(0)
    n_train = 18200
    # n_train = 500
    train_observation = np.array([]).reshape(0, 4 * window_size)
    test_observation = np.array([]).reshape(0, 4 * window_size)
    for i in range(n_train):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        train_observation = np.concatenate(
            (train_observation, temp_obs), axis=0
        )
    train_controls = controls[window_size : n_train + window_size].reshape(
        -1, 1
    )
    for i in range(n_train, n - window_size):
        temp_obs = np.array([]).reshape(1, 0)
        for j in range(window_size):
            temp_obs = np.concatenate(
                (temp_obs, observations[i + j, :].reshape(1, -1)), axis=1
            )
        test_observation = np.concatenate((test_observation, temp_obs), axis=0)
    test_controls = controls[n_train + window_size :].reshape(-1, 1)
    return train_observation, train_controls, test_observation, test_controls


def generate_repair_dataset(num_samples, bound, model):
    _, _, obs, ctrl = generateDataWindow(10)
    ctrl_pred = model.predict(obs)
    max_window_size = 600
    obs = obs[0:max_window_size]
    ctrl_pred = ctrl_pred[0:max_window_size].flatten()

    # violation_idx = np.argsort(np.abs(delta_u))[::-1]
    violation_idx = np.where(ctrl_pred >= bound[1])[0]
    temp = np.argsort(np.abs(ctrl_pred[violation_idx]))[::-1]
    violation_idx = violation_idx[temp]
    nonviolation_idx = np.where(ctrl_pred < bound[1])[0]
    if violation_idx.shape[0] == 0:
        nonviolation_idx = np.random.choice(
            nonviolation_idx, size=num_samples, replace=False
        )
        return obs[nonviolation_idx], ctrl[nonviolation_idx]
    else:
        violation_idx = np.random.choice(
            violation_idx, int(num_samples * 0.75), replace=False
        )
        nonviolation_idx = np.random.choice(
            nonviolation_idx, size=int(num_samples * 0.25), replace=False
        )
        idx = np.concatenate((violation_idx, nonviolation_idx))
        return obs[idx], ctrl[idx]


def plot_test_data(model_orig, model_repair, bounds):
    _, _, obs, ctrl = generateDataWindow(10)
    pred_ctrls_orig = model_orig(obs, training=False)
    pred_ctrls_repair = model_repair(obs, training=False)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(ctrl, color="#167fb8", label="Reference")
    ax1.plot(pred_ctrls_orig, color="#1abd15", label="Original Predictions")
    ax1.plot(
        pred_ctrls_repair,
        color="#b81662",
        label="Repaired Predictions",
    )
    ax1.axhline(y=bounds[0], color="k", linestyle="dashed")  # upper bound
    ax1.axhline(y=bounds[1], color="k", linestyle="dashed")  # lower bound
    ax1.set_ylabel("Ankle Angle Control (rad)")
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim([0, 500])
    ax1.legend()

    err_orig = np.abs(ctrl - pred_ctrls_orig)
    err_repair = np.abs(ctrl - pred_ctrls_repair)
    ax2.plot(err_orig, color="#1abd15")
    ax2.plot(err_repair, color="#b81662")
    ax2.grid(alpha=0.5, linestyle="dashed")
    ax2.set_ylabel("Ankle Angle Control Error (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim([0, 500])

    plt.show()