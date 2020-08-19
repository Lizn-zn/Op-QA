# -*- coding: utf-8 -*-
from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import torchvision as tv
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import torch.optim as optim

import math

import time

import sys
sys.path.append('../data')
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

from torch.utils.data import TensorDataset

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7,8,9'
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


def train_model(file_path='./model'):
    EPOCH = 100
    train_batch = 64
    test_batch = 256

    net = tv.models.inception_v3(
        pretrained=True, transform_input=True)
    net.aux_logits = False
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-5, momentum=0.9)
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # test_loss, test_acc = test(data_loader, net, criterion, use_cuda)

    valid_acc = []
    test_acc = []

    for epoch in range(EPOCH):
        correct = 0
        total = 0
        for i, data in enumerate(data_loader):
            if i < 200:
                net.train()
                inputs, labels = data
                inputs, labels = (inputs).cuda(), labels.cuda()
                # grad = 0
                optimizer.zero_grad()
                # forward + backward
                outputs = net(inputs)
                ce_loss = criterion(outputs, labels)


                loss = ce_loss

                loss.backward()
                optimizer.step()

                # print loss per batch_size
                print('The epoch %d, iteration %d, loss: %.03f'
                     % (epoch + 1, i + 1, loss.item()))
                # print acc per epoch
                continue
            if i == 200:
                net.eval()
                correct = 0
                total = 0
                for k, data in enumerate(data_loader):
                    if(k >= 200):
                        break
                    images, labels = data
                    images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('The %d epoch, acc is：%d%%' %
                      (epoch + 1, (100 * correct / total)))
                valid_acc.append(100 * correct / total)
                correct = 0
                total = 0
            if i >= 200 and i <= 400:
                net.eval()
                with torch.no_grad():
                    images, labels = data
                    images, labels = images.cuda(), labels.cuda()
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
            if i > 400:
                break 
        print('The %d epoch, acc is：%d%%' %
                          (epoch + 1, (100 * correct / total)))
            
        test_acc.append(100 * correct / total)

    np.savetxt('val_acc.csv', valid_acc, delimiter=',')
    np.savetxt('test_acc.csv', test_acc, delimiter=',')
    torch.save(net.state_dict(), 'fine_tune.pth')


if __name__ == "__main__":

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.405],
                                     std=[0.229, 0.224, 0.225])
    transforms = transforms.Compose([
        transforms.Resize(136),
        transforms.Pad(padding=120, padding_mode='edge'),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize,
    ])

    imagenet_data = tv.datasets.ImageNet(
        'val-data/', split='val', download=False, transform=transforms)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=25, shuffle=False,
                                              num_workers=4)


    train_model()
