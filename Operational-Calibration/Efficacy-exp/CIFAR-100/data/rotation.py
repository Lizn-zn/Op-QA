# -*- coding: utf-8 -*-
from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import torchvision as tv
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import PIL

import math

import time

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()

# transform = transforms.Compose([transforms.RandomRotation(30, resample=False, expand=True, center=None), transforms.ToTensor()])


transform = transforms.Compose(
    [
#        transforms.RandomRotation(30),
#        transforms.ColorJitter(brightness=0.5),
#        transforms.RandomAffine(10, (0.3, 0.3) ,fillcolor=(127,127,127)),
        transforms.RandomCrop(28),
        transforms.Pad(padding=4, padding_mode='edge'),
#        transforms.Resize(32),
#        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ]
)
dataset = ds.CIFAR100(root='.', train=False, transform=transform,
                      target_transform=None, download=True)
test_loader = DataLoader(
    dataset=dataset, batch_size=5000, shuffle=True, num_workers=4)
for i, data_set in enumerate(test_loader):
    if(i == 0):
        torch.save(data_set, 'operational.pt')
    elif(i == 1):
        torch.save(data_set, 'testing.pt')
    else:
        break
