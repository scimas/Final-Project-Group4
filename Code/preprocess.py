import os
import numpy as np
import pandas as pd
from skimage.transform import rotate


def load_data():
    """
    Read the data files, normalize the features and return them as train and test numpy arrays.
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    X_train = np.load(os.path.join(data_dir, "processed_images.npy"))
    X_train = X_train / 255
    y_train = np.load(os.path.join(data_dir, "processed_labels.npy"))

    X_test = np.load(os.path.join(data_dir, "test_images.npy"))
    X_test = X_test / 255
    y_test = np.load(os.path.join(data_dir, "test_labels.npy"))

    return X_train, X_test, y_train, y_test


def labels():
    text = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    return text


def augment_data():
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    df = pd.read_csv(os.path.join(data_dir, "sign_mnist_train.csv"))
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28)
    y = df.iloc[:, 0].values
    new_images = []
    new_labels = []
    for i, im in enumerate(X):
        # Pad image with zeros
        n_img = np.pad(im, 5, mode="constant", constant_values=0)
        new_images.append(n_img)
        new_labels.append(y[i])
        # Flip image horizontally
        new_images.append(n_img[:, ::-1])
        new_labels.append(y[i])
        # Rotate original image CCW
        new_images.append(rotate(im, 30, resize=True))
        new_labels.append(y[i])
        # Rotate original image CW
        new_images.append(rotate(im, -30, resize=True))
        new_labels.append(y[i])
        # Rotate flipped image CCW
        new_images.append(rotate(im[:, ::-1], 30, resize=True))
        new_labels.append(y[i])
        # Rotate flipped image CW
        new_images.append(rotate(im[:, ::-1], -30, resize=True))
        new_labels.append(y[i])
    
    new_images = np.array(new_images).reshape(-1, 1, 38, 38)
    new_labels = np.array(new_labels)
    np.save(os.path.join(data_dir, "processed_images.npy"), new_images, allow_pickle=False)
    np.save(os.path.join(data_dir, "processed_labels.npy"), new_labels, allow_pickle=False)

    df = pd.read_csv(os.path.join(data_dir, "sign_mnist_test.csv"))
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28)
    y = df.iloc[:, 0].values
    new_images = []
    for i, im in enumerate(X):
        new_images.append(np.pad(im, 5, mode="constant", constant_values=0))
    new_images = np.array(new_images).reshape(-1, 1, 38, 38)
    np.save(os.path.join(data_dir, "test_images.npy"), new_images, allow_pickle=False)
    np.save(os.path.join(data_dir, "test_labels.npy"), y, allow_pickle=False)


def augment_image(X):
    return np.pad(X, 5, mode="constant", constant_values=0).reshape(1, 38, 38)
