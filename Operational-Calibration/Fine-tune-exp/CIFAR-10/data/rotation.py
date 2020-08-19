# -*- coding: utf-8 -*-
from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import torchvision as tv
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import math

import time

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()

# transform = transforms.Compose([transforms.RandomRotation(30, resample=False, expand=True, center=None), transforms.ToTensor()])


transform = transforms.Compose(
    [
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]
)
test_set = ds.CIFAR10(root='.', train=False, transform=transform,
                      target_transform=None, download=True)


for i, data_set in enumerate(test_set):
    if(i == 0):
        torch.save(data_set, 'valid.pt')
    elif(i == 1):
        torch.save(data_set, 'testing.pt')
    else:
        break
