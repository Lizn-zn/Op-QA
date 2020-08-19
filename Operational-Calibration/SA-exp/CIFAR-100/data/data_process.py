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
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
                             # (0.2023, 0.1994, 0.2010))
    ]
)

# test_set = torch.load('operational_data.pt')
# test_loader = DataLoader(dataset=test_set,
#                          batch_size=5000,
#                          shuffle=True,
#                          num_workers=1)

# for batch_idx, (inputs, targets) in enumerate(test_loader):
#     if batch_idx == 0:
#         torch.save((inputs, targets), 'valid.pt')
#     elif batch_idx == 1:
#         torch.save((inputs, targets), 'test.pt')
#     else:
#       break


dataset = tv.datasets.CIFAR10('./', train=False, transform=transform)
test_loader = DataLoader(
    dataset=dataset, batch_size=5000, shuffle=True, num_workers=1)

for batch_idx, (inputs, targets) in enumerate(test_loader):
    if batch_idx == 0:
        torch.save((inputs, targets), 'training.pt')
