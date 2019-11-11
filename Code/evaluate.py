import os

import numpy as np
import preprocess
import torch

from modelling import Classifier, MyDataset
from sklearn.metrics import cohen_kappa_score, f1_score, make_scorer
from torch.utils.data import DataLoader


def score_func(y_true, y_pred):
    ckscr = cohen_kappa_score(y_true, y_pred)
    f1scr = f1_score(y_true, y_pred, average="macro")
    print("Cohen Kappa Score:", ckscr)
    print("F1 Score:         ", f1scr)
    return (ckscr + f1scr) / 2


X_train, X_test, y_train, y_test = preprocess.load_data()
test_data = MyDataset(X_test, y_test)
test_loader = DataLoader(
    test_data, batch_size=128, shuffle=False
)
base_dir = os.getcwd()
model_path = os.path.join(base_dir, "Code", "model", "sign_model.pth")
my_classifier = Classifier()
my_classifier.load_state_dict(torch.load(model_path))
my_classifier.cuda()

preds = []
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        pred = my_classifier(images)
        pred = torch.argmax(pred, axis=1).cpu().numpy().tolist()
        preds.extend(pred)

preds = np.array(preds)
print("Average:", score_func(y_test, preds))
