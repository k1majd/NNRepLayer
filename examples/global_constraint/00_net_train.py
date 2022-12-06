import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


def loadData(name_csv):
    with open(name_csv) as csv_file:
        data = np.asarray(
            list(csv.reader(csv_file, delimiter=",")), dtype=np.float32
        )
    return data


def generateDataWindow(window_size):
    Dfem = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_1.csv"
    )
    Dtib = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_2.csv"
    )
    Dfut = loadData(
        os.path.dirname(os.path.realpath(__file__)) + "/data/GeoffFTF_3.csv"
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


def buildModelWindow(num_inputs):
    layer_size1 = 32
    layer_size2 = 32
    layer_size3 = 32
    model = keras.Sequential(
        [
            keras.layers.Dense(
                layer_size1, activation=tf.nn.relu, input_shape=[num_inputs]
            ),
            keras.layers.Dense(layer_size2, activation=tf.nn.relu),
            keras.layers.Dense(layer_size3, activation=tf.nn.relu),
            keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=[tf.keras.losses.MeanAbsoluteError()],
        metrics=["accuracy"],
    )
    model.summary()
    architecture = [num_inputs, layer_size1, layer_size2, layer_size3, 1]
    filepath = os.path.dirname(os.path.realpath(__file__)) + "/tf_logs/check_points/model1"
    tf_callback = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath,
            monitor="val_loss",
            verbose=0,
            save_best_only=True,
            save_weight_only=False,
            mode="auto",
            save_freq="epoch",
            options=None,
        ),
        keras.callbacks.TensorBoard(log_dir=os.path.dirname(os.path.realpath(__file__)) + "/tf_logs"),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]

    return model, tf_callback, architecture


def plotTestData(model, test_obs, test_ctrls):
    pred_ctrls = model(test_obs, training=False)

    plt.figure(1)
    plt.plot(test_ctrls, color="#173f5f")
    plt.plot(pred_ctrls, color=[0.4705, 0.7921, 0.6470])
    plt.grid(alpha=0.5, linestyle="dashed")
    plt.ylabel("Ankle Angle Control [deg]")
    plt.xlabel("Sample [k]")
    plt.xlim([0, 500])
    plt.show()


if __name__ == "__main__":
    path_write = os.path.dirname(os.path.realpath(__file__)) + "/model_orig"
    if not os.path.exists(path_write):
        os.makedirs(path_write)

    # Train window model
    train_obs, train_ctrls, test_obs, test_ctrls = generateDataWindow(10)
    ctrl_model_orig, callback, architecture = buildModelWindow(
        train_obs.shape[1],
        )
    ctrl_model_orig.fit(
        train_obs,
        train_ctrls,
        validation_data=(test_obs, test_ctrls),
        batch_size=15,
        epochs=20,
        use_multiprocessing=True,
        verbose=1,
        shuffle=False,
        callbacks=callback,
    )
    keras.models.save_model(
        ctrl_model_orig,
        os.path.dirname(os.path.realpath(__file__))
        + "/model_orig",
        overwrite=True,
        include_optimizer=False,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    )
    print("saved: model")
    plotTestData(ctrl_model_orig, test_obs, test_ctrls)
