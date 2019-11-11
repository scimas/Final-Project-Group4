import torch

from torch import nn
from torch.utils.data import Dataset

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleDict()
        self.layers["conv1"] = nn.ModuleList(
            [
                # 38 x 38 x 1 -> 36 x 36 x 12
                nn.Conv2d(1, 16, 3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                # 36 x 36 x 16 -> 18 x 18 x 16
                nn.MaxPool2d(2),
            ]
        )
        self.layers["conv2"] = nn.ModuleList(
            [
                # 18 x 18 x 16 -> 16 x 16 x 8
                nn.Conv2d(16, 8, 3),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                # 16 x 16 x 8 -> 8 x 8 x 8
                nn.MaxPool2d(2),
            ]
        )
        self.layers["fc1"] = nn.ModuleList(
            [
                nn.Flatten(),
                # 8 x 8 x 8 -> 128
                nn.Linear(8 * 8 * 8, 128),
                nn.ReLU(),
            ]
        )
        self.layers["fc2"] = nn.ModuleList(
            [
                # 128 -> 26
                nn.Linear(128, 26),
            ]
        )
    

    def forward(self, X):
        for name, layerlist in self.layers.items():
            for layer in layerlist:
                X = layer(X)

        return X


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().cuda()
        self.y = torch.from_numpy(y).cuda()
    

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

    def __len__(self):
        return self.X.shape[0]
