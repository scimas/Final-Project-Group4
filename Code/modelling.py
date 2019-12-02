import os

import numpy as np
import torch

from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
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


class Committee(nn.Module):
    def __init__(self):
        super(Committee, self).__init__()
        base_dir = os.getcwd()
        model_name = "GoogleNet"
        self.my_classifier1 = get_model(model_name)
        model1_path = os.path.join(base_dir, "Code", "best_models", "model_09", "sign_model.pth")
        self.my_classifier1.load_state_dict(torch.load(model1_path))
        for param in self.my_classifier1.parameters():
            param.requires_grad = False
        self.my_classifier1.to(device)
        self.my_classifier1.eval()
        model_name = "ResNet"
        self.my_classifier2 = get_model(model_name)
        model2_path = os.path.join(base_dir, "Code", "best_models", "model_08", "sign_model.pth")
        self.my_classifier2.load_state_dict(torch.load(model2_path))
        for param in self.my_classifier2.parameters():
            param.requires_grad = False
        self.my_classifier2.to(device)
        self.my_classifier2.eval()
        self.fc = nn.Linear(26 * 2, 26)
    
    def forward(self, X):
        pred1 = self.my_classifier1(X)
        pred2 = self.my_classifier2(X)
        return self.fc(torch.cat(
            (pred1, pred2), dim=1
        ))
