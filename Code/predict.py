import io
import os
import sys, select
import time

import numpy as np
import preprocess
import requests
import torch

from matplotlib import pyplot as plt
from modelling import get_model, Committee
from skimage.io import imread
from skimage.transform import rotate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels = preprocess.labels()
# Load model
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "best_models", "model_10")
model_path = os.path.join(model_dir, "sign_model.pth")

my_committee = Committee()
my_committee.load_state_dict(torch.load(model_path))
my_committee.to(device)
my_committee.eval()
transform = preprocess.make_transform(mode="predict")

plt.ion()
camera_url = "http://scimas:abcd1234@161.253.112.73:8080/photo.jpg"
while True:
    r = requests.get(camera_url)
    f = io.BytesIO(r.content)
    im = imread(f)
    im = rotate(im, -90, resize=True)
    im = im[420:-420, :]
    im = transform(np.float32(im))
    im = im.view(1, 3, 224, 224)
    show_im = np.clip(im[0].numpy().transpose(1, 2, 0), 0, 1)
    
    plt.clf()
    plt.imshow(show_im)
    plt.draw()
    plt.pause(0.01)
    
    with torch.no_grad():
        pred = my_committee(im.to(device))
        pred = torch.argmax(pred, axis=1).cpu().item()
        print(pred, labels[pred])

    inp, o, e = select.select([sys.stdin], [], [], 1)
    if inp:
        if sys.stdin.readline().strip() == "q":
            plt.close()
            break
