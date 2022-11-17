import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision

from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
def process_image(image_path):
    re_size = 512
    crop_size = 448
    test_transform = transforms.Compose(
        [
            transforms.Resize((re_size, re_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img = Image.open(image_path)
    k = test_transform (img)
    return k
def predict(process_image):
    model = torch.load('max_acc.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    img = process_image.to(device)
    img = img.unsqueeze(0)
    print(img.shape)
    output = model(img)
    _, preds_tensor = torch.max(output, 1)

    file = open('./label.txt')
    dataMat = {}
    for line in file.readlines():
        # print(line)
        curLine = line.strip().split(" ")
        # print(curLine)
        dataMat[curLine[0]] = curLine[1]

    label_index = preds_tensor.cpu().numpy()
    print(dataMat[str(label_index[0] + 1)])
    return dataMat[str(label_index[0] + 1)]