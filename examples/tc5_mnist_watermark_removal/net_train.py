import os
import numpy as np
from tensorflow import keras


def load_model(json_path, model_path):
    # load json and create model
    json_file = open(json_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path)
    print("Loaded model from disk")
    return loaded_model


direc = os.path.dirname(os.path.realpath(__file__)) + "/raw_data"

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
wm_images = np.load(direc + "/wm_imgs.npy")  # watermark images
wm_labels = np.loadtxt(direc + "/wm_labels.txt", dtype="int32")  # watermark labels
wm_images = wm_images.reshape(
    wm_images.shape[0], wm_images.shape[1], wm_images.shape[2], 1
)


net_model = load_model(
    os.path.join(direc, "mnist_original_model.json"),
    os.path.join(direc, "mnist_original_model.h5"),
)
net_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001, name="Adam"),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
test_loss, test_acc = net_model.evaluate(x_test, y_test)
train_loss, train_acc = net_model.evaluate(x_train, y_train)
wm_loss, wm_acc = net_model.evaluate(wm_images, wm_labels)
