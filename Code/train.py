import os

import numpy as np
import preprocess
import torch

from modelling import MyDataset, get_model, Committee
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Optimizer learning rate
learning_rate = 1e-2
# Epochs
epochs = 15
# Validation loss early stopping patience
# Number of epochs
patience = 5

X_train, X_val, y_train, y_val, ws = preprocess.load_data()
train_data = MyDataset(X_train, y_train, preprocess.make_transform("train"))
train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available()
)
test_data = MyDataset(X_val, y_val, preprocess.make_transform("eval"))
test_loader = DataLoader(
    test_data, batch_size=512, shuffle=False, num_workers=8, pin_memory=torch.cuda.is_available()
)
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "model")
model_path = os.path.join(model_dir, "sign_model.pth")

my_committee = Committee()
my_committee.to(device)
criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(ws).to(device))
optimizer = optim.SGD(my_committee.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

min_val_loss = 1e10
loss_decreased = 0
for epoch in range(epochs):
    total_loss = 0
    my_committee.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = my_committee(images.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print("\rBatch: {:3d}".format(i), end="")
    print()
    print("Epoch: {:3d} Loss: {:10.3g}".format(epoch, total_loss/len(train_loader)), end=" ")
    with torch.no_grad():
        total_loss = 0
        my_committee.eval()
        for i, (images, labels) in enumerate(test_loader):
            pred = my_committee(images.to(device))
            loss = criterion(pred, labels.to(device))
            total_loss += loss.item()
        scheduler.step(total_loss/len(test_loader))
        print("Val loss: {:10.3g}".format(total_loss/len(test_loader)))
        if total_loss < min_val_loss:
            min_val_loss = total_loss
            loss_decreased = 0
            torch.save(my_committee.state_dict(), model_path)
        else:
            loss_decreased += 1
        if loss_decreased == patience:
            print("Validation loss didn't decrease for", patience, "epochs. Breaking early.")
            print("Best model has been saved.")
            break
optimizer.zero_grad()
