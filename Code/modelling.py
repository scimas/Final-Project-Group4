import torch
from torch import nn
from torch.utils.data import Dataset

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 38 x 38 x 1 -> 36 x 36 x 6
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.actv1 = nn.ReLU()
        # 36 x 36 x 6 -> 18 x 18 x 6
        self.pool1 = nn.MaxPool2d(2)
        # # 18 x 18 x 6 -> 16 x 16 x 12
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.actv2 = nn.ReLU()
        # # 16 x 16 x 12 -> 8 x 8 x 12
        self.pool2 = nn.MaxPool2d(2)
        # # 8 x 8 x 12 -> 26
        self.fc = nn.Linear(8 * 8 * 12, 26)
    

    def forward(self, X):
        X = self.pool1(self.actv1(self.conv1(X)))
        X = self.pool2(self.actv2(self.conv2(X)))
        X = X.view(-1, 8 * 8 * 12)
        X = self.fc(X)

        return X


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().cuda()
        self.y = torch.from_numpy(y).cuda()
    

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

    def __len__(self):
        return self.X.shape[0]
