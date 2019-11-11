import torch
import preprocess
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from modelling import Classifier, MyDataset

X_train, X_test, y_train, y_test = preprocess.load_data()
train_data = MyDataset(X_train, y_train)
train_loader = DataLoader(
    train_data, batch_size=128, shuffle=True
)
test_data = MyDataset(X_test, y_test)
labels = preprocess.labels()

my_classifier = Classifier()
my_classifier.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_classifier.parameters(), lr=10**-4)

for epoch in range(200):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = my_classifier(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())
