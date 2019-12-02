import os

import numpy as np
import preprocess
import torch

from modelling import MyDataset, get_model, Committee
from sklearn.metrics import cohen_kappa_score, f1_score, make_scorer
from torch.utils.data import DataLoader


def score_func(y_true, y_pred):
    ckscr = cohen_kappa_score(y_true, y_pred)
    f1scr = f1_score(y_true, y_pred, average="macro")
    print("Cohen Kappa Score:", ckscr)
    print("F1 Score:         ", f1scr)
    return (ckscr + f1scr) / 2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
alphabet = preprocess.labels()
X_train, X_te, y_train, y_te, train_ws = preprocess.load_data()
X_train = np.vstack((X_train, X_te))
y_train = np.hstack((y_train, y_te))
train_data = MyDataset(X_train, y_train, preprocess.make_transform(mode="eval"))
train_loader = DataLoader(
    train_data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True
)

X_test, y_test, test_ws = preprocess.load_test_data()
test_data = MyDataset(X_test, y_test, preprocess.make_transform(mode="eval"))
test_loader = DataLoader(
    test_data, batch_size=512, shuffle=False, num_workers=8, pin_memory=True
)
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "model")
model_path = os.path.join(model_dir, "sign_model.pth")

my_committee = Committee()
my_committee.load_state_dict(torch.load(model_path))
my_committee.to(device)
my_committee.eval()

preds = []
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        pred = my_committee(images.to(device))
        pred = torch.argmax(pred, axis=1).cpu().numpy().tolist()
        preds.extend(pred)

preds = np.array(preds)
print(alphabet[np.unique(preds)])
print("Average on test data:", score_func(y_test, preds))
print()

preds = []
with torch.no_grad():
    for i, (images, labels) in enumerate(train_loader):
        pred = my_committee(images.to(device))
        pred = torch.argmax(pred, axis=1).cpu().numpy().tolist()
        preds.extend(pred)

preds = np.array(preds)
print("Average on train data:", score_func(y_train, preds))
