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
#        transforms.CenterCrop(32),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# test_set = torch.load('operational_data.pt')
# test_loader = DataLoader(dataset=test_set,
#                          batch_size=5000,
#                          shuffle=True,
#                          num_workers=1)


dataset = tv.datasets.STL10('./', split='test', download=True, transform=transform)
test_loader = DataLoader(
    dataset=dataset, batch_size=5000, shuffle=True, num_workers=1)


for batch_idx, (inputs, targets) in enumerate(test_loader):
    if batch_idx == 0:
        print(inputs.shape)
        torch.save((inputs, targets), 'operational.pt')
    elif batch_idx == 1:
        print(inputs.shape)
        torch.save((inputs, targets), 'test.pt')
    else:
        break


# dataset = tv.datasets.stl10('./', train=False, transform=transform)
# test_loader = DataLoader(
#     dataset=dataset, batch_size=5000, shuffle=True, num_workers=1)

# for batch_idx, (inputs, targets) in enumerate(test_loader):
#     if batch_idx == 0:
#         torch.save((inputs, targets), 'training.pt')
