import io
import json
import os
import sys, select
import time

import cv2
import numpy as np
import preprocess
import requests
import torch

from best_models.model_01.modelling import Classifier, ConvSize
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import rotate

# # Start camera
# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
labels = preprocess.labels()
# Load model
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "best_models", "model_01")
model_path = os.path.join(model_dir, "sign_model.pth")

conv_sizes = []
with open(os.path.join(model_dir, "model_specification"), "r") as ms:
    spec = json.load(ms)
fc_sizes = spec["fc"]
for layer in spec["conv"]:
    conv_sizes.append(ConvSize(*layer))

my_classifier = Classifier(conv_sizes, fc_sizes)
my_classifier.load_state_dict(torch.load(model_path))
my_classifier.cuda()
my_classifier.eval()

# good, im = cam.read()
# time.sleep(1)

plt.ion()
camera_url = "http://scimas:password@10.0.0.81:8080/photo.jpg"
while True:
    # good, im = cam.read()
    # time.sleep(0.01)
    # good, im = cam.read()
    r = requests.get(camera_url)
    f = io.BytesIO(r.content)
    im = imread(f)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    im = rotate(im, 90, resize=True)
    im = im[420:-420, :]
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)
    
    plt.clf()
    plt.imshow(im, cmap="gray")
    plt.draw()
    plt.pause(0.01)
    
    im = im.reshape(1, 1, 28, 28)
    with torch.no_grad():
        pred = my_classifier(torch.from_numpy(im).float().cuda())
        pred = torch.argmax(pred, axis=1).cpu().item()
        print(pred, labels[pred])

    inp, o, e = select.select([sys.stdin], [], [], 1)
    if inp:
        if sys.stdin.readline().strip() == "q":
            plt.close()
            # cam.release()
            break
