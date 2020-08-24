# -*- coding: utf-8 -*-
from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import torchvision as tv
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from PIL import Image

transform = transforms.Compose(
    [
        # transforms.RandomRotation(30),
        transforms.ToTensor(),
    ]
)
t = transforms.ToPILImage()

dataset = tv.datasets.CIFAR10('./', train=True, transform=transform)

data_loader = DataLoader(
    dataset=dataset, batch_size=9, shuffle=True)

for image, label in data_loader:
    tv.utils.save_image(image, 'orig2.png', nrow=3)
    break

# im = Image.open('p/2008_000133.jpg')
# # im = Image.open('p/2009_002649.jpg')
# # im = Image.open('c/224_0003.jpg')
# # im = Image.open('c/246_0043.jpg')
# im = im.resize((224, 224))
# im.save('opt1.png')
