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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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


def train_model(file_path='./model'):
    EPOCH = 100
    train_batch = 32
    test_batch = 256

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]
    )

    train_loader, valid_loader = dp.load_imageclef_train(
        '../data/', 'p', train_batch, 'src')
    # test_loader = dp.load_imageclef_test('../data/', 'c', train_batch, 'src')
    test_loader = valid_loader

    model_dir = 'best_resnet_c.pth'
    net = tv.models.resnet50(pretrained=False)
    net.fc = nn.Linear(2048, 12)

    net.load_state_dict(torch.load(
        model_dir, map_location=torch.device('cpu')))
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(test_loader, net, criterion, use_cuda)

    ignored_params = list(map(id, net.fc.parameters()))
    base_params = filter(lambda p: id(
        p) not in ignored_params, net.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([{'params': base_params}, {
         'params': net.fc.parameters(), 'lr': 1e-3}], lr=1e-4)
    # optimizer = optim.Adam(net.parameters(), lr=1e-4)
    net.train()

    valid_acc = []
    test_acc = []
    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

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
        with torch.no_grad():
            correct = 0
            total = 0.0
            for data in train_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('The {!s} epoch, valid acc is：{!s}'.format
                  (epoch + 1, (correct / total)))
            valid_acc.append(correct / total)

            correct = 0
            total = 0.0
            for data in test_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('The {!s} epoch, test acc is：{!s}'.format
                  (epoch + 1, (correct / total)))
            test_acc.append(correct / total)

    np.savetxt('val_acc.csv', valid_acc, delimiter=',')
    np.savetxt('test_acc.csv', test_acc, delimiter=',')
    torch.save(net.state_dict(), 'fine_tune.pth')


if __name__ == "__main__":
    train_model()
