import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models


class MyDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y)
        self.transforms = transforms


    def __getitem__(self, idx):
        if self.transforms:
            return self.transforms(self.X[idx]), self.y[idx]
        else:
            return self.X[idx], self.y[idx]
    

    def __len__(self):
        return self.X.shape[0]


def get_model(model_name):
    model = None
    if model_name == "AlexNet":
        model = models.alexnet(pretrained=True)
    elif model_name == "VGG":
        model = models.vgg16(pretrained=True)
    elif model_name == "ResNet":
        model = models.resnet18(pretrained=True)
    elif model_name == "SqueezNet":
        model = models.squeezenet1_1(pretrained=True)
    elif model_name == "DenseNet":
        model = models.densenet161(pretrained=True)
    elif model_name == "Inception":
        model = models.inception_v3(pretrained=True)
    elif model_name == "GoogleNet":
        model = models.googlenet(pretrained=True)
    elif model_name == "ShuffleNet":
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == "MobileNet":
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == "ResNext":
        model = models.resnext101_32x8d(pretrained=True)
    elif model_name == "WResNet":
        model = models.wide_resnet101_2(pretrained=True)
    elif model_name == "MNASNet":
        model = models.mnasnet1_0(pretrained=True)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 26)

    return model
