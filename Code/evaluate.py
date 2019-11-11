import torch
import preprocess
import numpy as np
from torch.utils.data import DataLoader
from modelling import Classifier, MyDataset
from sklearn.metrics import cohen_kappa_score, f1_score, make_scorer

def score_func(y_true, y_pred):
    return (cohen_kappa_score(y_true, y_pred) + f1_score(y_true, y_pred, average="macro")) / 2

X_train, X_test, y_train, y_test = preprocess.load_data()
test_data = MyDataset(X_test, y_test)
test_loader = DataLoader(
    test_data, batch_size=128, shuffle=False
)
model_path = "model/sign_model.pth"
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
print(score_func(y_test, preds))
