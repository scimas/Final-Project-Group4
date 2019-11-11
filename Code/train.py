import os

import numpy as np
import preprocess
import torch

from modelling import Classifier, MyDataset
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Optimizer learning rate
learning_rate = 1e-4
# Epochs
epochs = 100
# Validation loss early stopping patience
# Number of epochs
patience = 5

X_train, X_test, y_train, y_test = preprocess.load_data()
train_data = MyDataset(X_train, y_train)
train_loader = DataLoader(
    train_data, batch_size=256, shuffle=True
)
test_data = MyDataset(X_test, y_test)
test_loader = DataLoader(
    test_data, batch_size=256, shuffle=False
)
labels = preprocess.labels()
base_dir = os.getcwd()
model_path = os.path.join(base_dir, "Code", "model", "sign_model.pth")

my_classifier = Classifier()
my_classifier.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_classifier.parameters(), lr=learning_rate)

min_val_loss = 10e10
loss_decreased = 0
for epoch in range(epochs):
    total_loss = 0
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
