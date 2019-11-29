import io
import os
import sys, select
import time

import numpy as np
import preprocess
import requests
import torch

from best_models.model_05.modelling import get_model
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import rotate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels = preprocess.labels()
# Load model
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "best_models", "model_05")
model_path = os.path.join(model_dir, "sign_model.pth")

with open(os.path.join(model_dir, "model_specification"), "r") as ms:
    model_name = ms.readline().strip()

my_classifier = get_model(model_name)
my_classifier.load_state_dict(torch.load(model_path))
my_classifier.to(device)
my_classifier.eval()
transform = preprocess.make_transform(mode="predict")

plt.ion()
camera_url = "http://scimas:abcd1234@10.0.0.81:8080/photo.jpg"
while True:
    r = requests.get(camera_url)
    f = io.BytesIO(r.content)
    im = imread(f)
    im = rotate(im, -90, resize=True)
    im = im[420:-420, :]
    im = transform(np.float32(im))
    im = im.view(1, 3, 224, 224)
    
    plt.clf()
    plt.imshow(im[0].numpy().transpose(1, 2, 0))
    plt.draw()
    plt.pause(0.01)
    
    with torch.no_grad():
        pred = my_classifier(im.to(device))
        pred = torch.argmax(pred, axis=1).cpu().item()
        print(pred, labels[pred])

    inp, o, e = select.select([sys.stdin], [], [], 1)
    if inp:
        if sys.stdin.readline().strip() == "q":
            plt.close()
            break
