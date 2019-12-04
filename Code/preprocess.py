import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms


def load_data():
    """
    Read the data files, normalize the features and return them as train and test numpy arrays.
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    if not os.path.exists(os.path.join(data_dir, "train_images.npy")):
        print("Data hasn't been preprocessed, possibly first run.")
        print("Please wait a few minutes.")
        augment_data()
        print("Done preprocessing, now loading.")
    X = np.load(os.path.join(data_dir, "train_images.npy"))
    y = np.load(os.path.join(data_dir, "train_labels.npy"))
    ws = np.load(os.path.join(data_dir, "train_weights.npy"))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y)

    return X_train, X_val, y_train, y_val, ws


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
    y = np.load(os.path.join(data_dir, "test_labels.npy"))
    ws = np.load(os.path.join(data_dir, "test_weights.npy"))

    return X, y, ws


def labels():
    text = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I",
        "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    return np.array(text)


def augment_data():
    """
    Read the raw data and store it in easily loadable formats.
    """
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    df = pd.read_csv(os.path.join(data_dir, "augmented.csv"))
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    X = X.reshape(-1, 28, 28, 1)
    ws = df["label"].value_counts()
    ws = ws.max() / ws
    ws.at[9] = ws.at[25] = 0
    train_weights = np.float32(np.array([ws[i] for i in range(26)]))
    
    np.save(os.path.join(data_dir, "train_images.npy"), X, allow_pickle=False)
    np.save(os.path.join(data_dir, "train_labels.npy"), y, allow_pickle=False)
    np.save(os.path.join(data_dir, "train_weights.npy"), train_weights, allow_pickle=False)

    df = pd.read_csv(os.path.join(data_dir, "sign_mnist_test.csv"))
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    X = X.reshape(-1, 28, 28, 1)
    ws = df["label"].value_counts()
    ws = ws.max() / ws
    ws.at[9] = ws.at[25] = 0
    test_weights = np.float32(np.array([ws[i] for i in range(26)]))
    
    np.save(os.path.join(data_dir, "test_images.npy"), X, allow_pickle=False)
    np.save(os.path.join(data_dir, "test_labels.npy"), y, allow_pickle=False)
    np.save(os.path.join(data_dir, "test_weights.npy"), test_weights, allow_pickle=False)


class ReplicateChannel(object):
    def __call__(self, image):
        return image.view(-1).repeat(3).view(3, 224, 224)


def make_transform(mode="train"):
    """
    Various transforms used for training, evaluation and prediction
    """
    resize = transforms.Resize((224, 224), interpolation=Image.LANCZOS)
    hflip = transforms.RandomHorizontalFlip()
    rotate = transforms.RandomRotation(10, resample=Image.BICUBIC)
    jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)
    toTensor = transforms.ToTensor()
    replicate = ReplicateChannel()
    normalization = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )

    mods = None
    if mode == "train":
        toPIL = transforms.ToPILImage()
        mods = transforms.Compose([
            toTensor, toPIL, resize, hflip, rotate, jitter, toTensor, replicate, normalization
        ])
    elif mode == "eval":
        toPIL = transforms.ToPILImage()
        mods = transforms.Compose([
            toTensor, toPIL, resize, toTensor, replicate, normalization
        ])
    elif mode == "predict":
        toPIL = transforms.ToPILImage(mode="RGB")
        resize2 = transforms.Resize((28, 28), interpolation=Image.LANCZOS)
        mods = transforms.Compose([
            toTensor, toPIL, resize2, resize, toTensor, normalization
        ])

    return mods
