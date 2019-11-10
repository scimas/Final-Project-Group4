import numpy as np
import pandas as pd
from skimage.transform import rotate


def load_data():
    """
    Read the data files, normalize the features and return them as train and test numpy arrays.
    """
    X_train = np.load("../data/processed_images.npy")
    X_train = X_train / 255
    y_train = np.load("../data/processed_labels.npy")

    df = pd.read_csv("../data/sign_mnist_test.csv")
    X_test = df.iloc[:, 1:].values
    X_test = X.reshape(-1, 28, 28) / 255
    y_test = df.iloc[:, 0].values

    return X_train, X_test, y_train, y_test


def labels():
    text = [
        "A", "B", "C", "D", "E", "F", "G", "H",
        "I", "K", "L", "M", "N", "O", "P", "Q",
        "R", "S", "T", "U", "V", "W", "X", "Y"
    ]

    return text


def augment_data():
    df = pd.read_csv("../data/sign_mnist_train.csv")
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
        if i % 100 == 0:
            print(i)
    
    new_images = np.array(new_images)
    new_labels = np.array(new_labels)
    np.save("../data/processed_images.npy", new_images, allow_pickle=False)
    np.save("../data/processed_labels.npy", new_labels, allow_pickle=False)
