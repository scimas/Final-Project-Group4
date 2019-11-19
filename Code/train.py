import json
import os

import numpy as np
import preprocess
import torch

from modelling import MyDataset, get_model
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Optimizer learning rate
learning_rate = 1e-4
# Epochs
epochs = 10
# Validation loss early stopping patience
# Number of epochs
patience = 1

X_train, X_val, y_train, y_val = preprocess.load_data()
train_data = MyDataset(X_train, y_train, preprocess.make_transform())
train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)
test_data = MyDataset(X_val, y_val, preprocess.make_transform())
test_loader = DataLoader(
    test_data, batch_size=512, shuffle=False
)
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "model")
model_name = "ResNet"
model_path = os.path.join(model_dir, "sign_model.pth")
with open(os.path.join(model_dir, "model_specification"), "w") as ms:
    ms.write(model_name)

my_classifier = get_model(model_name)
my_classifier.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_classifier.parameters(), lr=learning_rate)

min_val_loss = 10e10
loss_decreased = 0
for epoch in range(epochs):
    total_loss = 0
    my_classifier.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = my_classifier(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("Epoch: {:3d} Loss: {:6.4f}".format(epoch, total_loss/len(train_loader)), end=" ")
    with torch.no_grad():
        total_loss = 0
        my_classifier.eval()
        for i, (images, labels) in enumerate(test_loader):
            pred = my_classifier(images.to(device))
            loss = criterion(pred, labels.to(device))
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
