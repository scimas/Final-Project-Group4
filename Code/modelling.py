import numpy as np
import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models


class MyDataset(Dataset):
    """
    Extend pytorch Dataset for my numpy dataset image loading
    """
    def __init__(self, X, y, transform=None):
        self.X = np.float32(X)
        self.y = torch.from_numpy(y)
        self.transform = transform


    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.X[idx], self.y[idx]
    

    def __len__(self):
        return self.X.shape[0]


def get_model(model_name):
    """
    Get specific modified pre-trained models by name
    """
    model = None
    if model_name == "AlexNet":
        model = models.alexnet(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "VGG":
        model = models.vgg16(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "ResNet":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "SqueezNet":
        model = models.squeezenet1_1(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "DenseNet":
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 26)
    elif model_name == "Inception":
        model = models.inception_v3(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "GoogleNet":
        model = models.googlenet(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "ShuffleNet":
        model = models.shufflenet_v2_x1_0(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "MobileNet":
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "ResNext":
        model = models.resnext101_32x8d(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "WResNet":
        model = models.wide_resnet101_2(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)
    elif model_name == "MNASNet":
        model = models.mnasnet1_0(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 26)

    return model
