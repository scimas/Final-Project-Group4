import json
import os

import numpy as np
import preprocess
import torch

from modelling import ConvSize, Classifier, MyDataset
from torch import nn
from torch import optim
from torch.utils.data import DataLoader


def randomize_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        m.reset_parameters()


# Optimizer learning rate
learning_rate = 1e-4
# Epochs
epochs = 200
# Validation loss early stopping patience
# Number of epochs
patience = 10

X_train, X_val, y_train, y_val = preprocess.load_data()
train_data = MyDataset(X_train, y_train)
train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)
test_data = MyDataset(X_val, y_val)
test_loader = DataLoader(
    test_data, batch_size=512, shuffle=False
)
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "model")
model_path = os.path.join(model_dir, "sign_model.pth")

conv_sizes = [
    ConvSize(64, 3, 0, 2, 0), # 28 -> 26 -> 13
    ConvSize(64, 3, 0, 2, 0), # 13 -> 11 -> 5
    ConvSize(64, 3, 0, 2, 0), # 5 -> 3 -> 1
]
fc_sizes = [512, 64, 26]
with open(os.path.join(model_dir, "model_specification"), "w") as ms:
    json.dump({"conv": conv_sizes, "fc": fc_sizes}, ms)

my_classifier = Classifier(conv_sizes, fc_sizes)
my_classifier.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_classifier.parameters(), lr=learning_rate)

min_val_loss = 10e10
loss_decreased = 0
for epoch in range(epochs):
    total_loss = 0
    my_classifier.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = my_classifier(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch: {:3d} Loss: {:6.4f}".format(epoch, total_loss/len(train_loader)), end=" ")
    with torch.no_grad():
        total_loss = 0
        my_classifier.eval()
        for i, (images, labels) in enumerate(test_loader):
            pred = my_classifier(images)
            loss = criterion(pred, labels)
            total_loss += loss.item()
        print("Val loss: {:6.4f}".format(total_loss/len(test_loader)))
        if total_loss < min_val_loss:
            min_val_loss = total_loss
            loss_decreased = 0
            torch.save(my_classifier.state_dict(), model_path)
        else:
            loss_decreased += 1
        if loss_decreased == patience:
            print("Validation loss didn't decrease for", patience, "epochs. Breaking early.")
            print("Best model has been saved.")
            break
optimizer.zero_grad()
