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

from torch.utils.data import TensorDataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

import os
CUDA_VISIBLE_DEVICES = 1

device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")

import sys
sys.path.append('../data/')
import data_process as dp


def test(testloader, model, criterion, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)



if __name__ == "__main__":
    train_batch = 32
    test_batch = 256
    test_loader = dp.load_imageclef_test('../data/', 'p', train_batch, 'src')
    # test_loader = dp.load_imageclef_test('../data/', 'c', train_batch, 'src')
    # test_loader = dp.load_imageclef_test('../data/', 'i', train_batch, 'src')
    # loss function and optimization
    model_dir = 'best_resnet_p.pth'
    net = tv.models.resnet50(pretrained=False)
    net.fc = nn.Linear(2048, 12)

    net.load_state_dict(torch.load(
        model_dir, map_location=torch.device('cpu')))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(test_loader, net, criterion, use_cuda)
