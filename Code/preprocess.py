import numpy as np
import pandas as pd
from skimage.transform import rotate
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        if i % 1000 == 0:
            print(i)
    
    new_images = np.array(new_images).reshape(-1, 1, 38, 38)
    new_labels = np.array(new_labels)
    np.save("../data/processed_images.npy", new_images, allow_pickle=False)
    np.save("../data/processed_labels.npy", new_labels, allow_pickle=False)

    df = pd.read_csv("../data/sign_mnist_test.csv")
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28)
    y = df.iloc[:, 0].values
    new_images = []
    for i, im in enumerate(X):
        new_images.append(np.pad(im, 5, mode="constant", constant_values=0))
    new_images = np.array(new_images).reshape(-1, 1, 38, 38)
    np.save("../data/test_images.npy", new_images, allow_pickle=False)
    np.save("../data/test_labels.npy", y, allow_pickle=False)

def augment_image(X):
    return np.pad(X, 5, mode="constant", constant_values=0).reshape(1, 38, 38)

# augment_data()