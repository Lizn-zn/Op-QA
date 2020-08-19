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

from torch.nn import functional as F

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from torch.utils.data import TensorDataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,8,9'
use_cuda = torch.cuda.is_available()


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

if __name__ == '__main__':
    x_test, y_test = torch.load('original.pt')
    test_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=256,
                             shuffle=True,
                             num_workers=2)

    # load original model
    import sys
    sys.path.append('../model/')
    import resnet
    net = resnet.ResNet18()
    # net = torch.nn.DataParallel(net).cuda()
    state_dict = torch.load('../model/model_best.pth')
    # net.load_state_dict(state_dict)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
         name = k.replace('module.', '') 
         new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    # net.load_state_dict(state_dict)
    net.eval()
    net.cuda()


    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(
        test_loader, net, criterion, use_cuda)
    
    x_test, y_test = torch.load('operational.pt')
    # load operational test data
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    test_set = TensorDataset(x_test, y_test)
    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
    )


    x_test = np.array([])
    y_test = np.array([])
    # load operational test data
    for inputs, targets in test_loader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(
            # inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_test = np.append(x_test, outputs.cpu().detach().numpy())
        y_test = np.append(y_test, targets.cpu().detach().numpy())
    x_test = torch.Tensor(x_test.reshape(-1, 512))
    y_test = torch.Tensor(y_test.reshape(-1))
    x_test = x_test.cuda()
    pred = net.classifier(x_test)
    conf, pred = torch.max(F.softmax(pred,dim=1),1)
    pred = pred.cpu().detach().numpy()
    conf = conf.cpu().detach().numpy()
    indicate = pred == y_test.detach().numpy()
    print('brier score is {}'.format(np.mean(np.square(conf - indicate))))	
