import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch
from sklearn.metrics import cohen_kappa_score, f1_score
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
LR = 0.1
N_EPOCHS = 15
BATCH_SIZE = 1000

def load_data():
    """
    Read the data files, normalize the features and return them as train and test numpy arrays.
    """
    X_train = np.load("../data/processed_images.npy")
    X_train = X_train / 255
    y_train = np.load("../data/processed_labels.npy")

    X_test = np.load("../data/test_images.npy")
    X_test = X_test / 255
    y_test = np.load("../data/test_labels.npy")

    return X_train, X_test, y_train, y_test

def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)  # output (n_examples, 20, 36, 36)
        self.convnorm1 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2) # output (n_examples, 20, 18, 18)
        self.conv2 = nn.Conv2d(20, 20, 5) # output (n_examples, 20, 14, 14)
        self.convnorm2 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2) # output (n_examples, 20, 7, 7)
        self.linear1 = nn.Linear(20*7*7, 100)   # flatten input to (n_examples, 20*7*7) and output (n_examples, 100)
        self.norm3 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 24)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.norm3(self.act(self.linear1(x.view(len(x), -1))))
        return self.linear2(x)

X_train, X_test, y_train, y_test = load_data()
X_train, X_test, y_train, y_test = torch.tensor(X_train).float().to(device), \
                                   torch.tensor(X_test).float().to(device), \
                                   torch.tensor(y_train).long().to(device), \
                                   torch.tensor(y_test).long().to(device)
y_train = torch.tensor([y if y<=8 else y-1 for y in y_train]).to(device)
y_test = torch.tensor([y if y<=8 else y-1 for y in y_test]).to(device)
print(X_train.shape)

model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("Starting training loop...")
for epoch in range(N_EPOCHS):
    loss_train = 0
    model.train()

    pred_train = []
    for batch in range(len(X_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(X_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        pred_train += list(torch.argmax(logits, axis=1).cpu().numpy().reshape(-1))

    acc_train = accuracy_score(y_train.cpu(), np.array(pred_train))*100

    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()
        y_pred = torch.argmax(y_test_pred, axis=1).cpu().numpy().reshape(-1)

    print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f} - F1 {:.2f} - Cohen Kappa {:.2f}".format(
        epoch, loss_train / BATCH_SIZE, acc_train, loss_test, acc(X_test, y_test), f1_score(y_test.cpu(), y_pred, average='macro'), \
        cohen_kappa_score(y_test.cpu(), y_pred)))

