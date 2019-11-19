import io
import os
import sys, select
import time

import cv2
import numpy as np
import preprocess
import requests
import torch

from best_models.model_03.modelling import get_model
from matplotlib import pyplot as plt
from skimage.transform import rotate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Start camera
# cam = cv2.VideoCapture(0)
# cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
labels = preprocess.labels()
# Load model
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "Code", "best_models", "model_01")
model_path = os.path.join(model_dir, "sign_model.pth")

with open(os.path.join(model_dir, "model_specification"), "r") as ms:
    model_name = ms.readline()

my_classifier = get_model(model_name)
my_classifier.load_state_dict(torch.load(model_path))
my_classifier.to(device)
my_classifier.eval()
transforms = preprocess.make_transform(mode="predict")

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
    im = imread(f) / 255
    im = rotate(im, 90, resize=True)
    im = im[420:-420, :]
    im = transforms(im)
    
    plt.clf()
    plt.imshow(im.numpy(), cmap="gray")
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
            # cam.release()
            break
