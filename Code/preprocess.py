import os

import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torchvision import transforms


def load_data():
    """
    Read the data files, normalize the features and return them as train and test numpy arrays.
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(os.path.join(data_dir, "processed_images.npy")):
        print("Data hasn't been preprocessed, possibly first run.")
        print("Please wait a few minutes.")
        augment_data()
        print("Done preprocessing, now loading.")
    X = np.load(os.path.join(data_dir, "processed_images.npy"))
    X = X / 255
    y = np.load(os.path.join(data_dir, "processed_labels.npy"))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y)

    return X_train, X_val, y_train, y_val


def load_test_data():
    """
    Read the data files, normalize the features and return them as train and test numpy arrays.
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(os.path.join(data_dir, "test_images.npy")):
        print("Data hasn't been preprocessed, possibly first run.")
        print("Please wait a few minutes.")
        augment_data()
        print("Done preprocessing, now loading.")
    X = np.load(os.path.join(data_dir, "test_images.npy"))
    X = X / 255
    y = np.load(os.path.join(data_dir, "test_labels.npy"))

    return X, y


def labels():
    text = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    return np.array(text)


def augment_data():
    ros = RandomOverSampler()
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    df = pd.read_csv(os.path.join(data_dir, "sign_mnist_train.csv"))
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    X, y = ros.fit_sample(X, y)
    X = X.reshape(-1, 1, 28, 28)
    
    np.save(os.path.join(data_dir, "processed_images.npy"), X, allow_pickle=False)
    np.save(os.path.join(data_dir, "processed_labels.npy"), y, allow_pickle=False)

    df = pd.read_csv(os.path.join(data_dir, "sign_mnist_test.csv"))
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    X, y = ros.fit_sample(X, y)
    X = X.reshape(-1, 1, 28, 28)
    np.save(os.path.join(data_dir, "test_images.npy"), X, allow_pickle=False)
    np.save(os.path.join(data_dir, "test_labels.npy"), y, allow_pickle=False)


def make_transform():
    toPIL = transforms.ToPILImage()
    resize = transforms.Resize((3, 224, 224), interpolation=Image.LANCZOS)
    hflip = transforms.RandomHorizontalFlip()
    rotate = transforms.RandomRotation(10, resample=Image.BICUBIC)
    toTensor = transforms.ToTensor()
    normalization = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )

    transforms = transforms.Compose([
        toPIL, resize, hflip, rotate, toTensor, normalization
    ])

    return transforms
