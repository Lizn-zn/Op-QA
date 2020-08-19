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
    train_batch = 32
    test_batch = 256

    t = transforms.Compose(
        [    
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
        ]
    )

    x_train, y_train = torch.load('../data/operational.pt')
    x_train = torch.FloatTensor(x_train)[0:100]
    y_train = torch.LongTensor(y_train)[0:100]
    train_set = TensorDataset(x_train, y_train)

    # train batch
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch,
        shuffle=True,
    )

    x_test, y_test = torch.load('../data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    test_set = TensorDataset(x_test, y_test)
    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch,
        shuffle=False,
    )

    # loss function and optimization
    import resnet
    net = resnet.ResNet18()
    model_dir = './model_best.pth'
    # BackBone.load_state_dict(torch.load(model_dir, map_location='cpu'))
    # net.load_state_dict(state_dict)
    state_dict = torch.load(model_dir)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    net.cuda()


    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(test_loader, net, criterion, use_cuda)

    test_loss, test_acc = test(train_loader, net, criterion, use_cuda) 

    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)
    net.train()

    valid_acc = []
    test_acc = []
    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            inputs, labels = data
#            inputs, labels = inputs.cuda(), labels.cuda()
#            temp = []
#            for k in range(inputs.shape[0]):
#                temp.append(t(inputs[k]).detach().numpy())
#            temp = np.array(temp)
#            inputs = torch.Tensor(temp.reshape(-1,3,32,32))
            inputs, labels = inputs.cuda(), labels.cuda()

            # grad = 0
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            ce_loss = criterion(outputs, labels)

            # reg_loss = torch.tensor(0.)
            # for param in net.parameters():
            # reg_loss += torch.norm(param)

            loss = ce_loss

            loss.backward()
            optimizer.step()

            # print loss per batch_size
            print('The epoch %d, iteration %d, loss: %.03f'
                  % (epoch + 1, i + 1, loss.item()))
        # print acc per epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in train_loader:
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
            for data in test_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('The %d epoch, acc is：%d%%' %
                  (epoch + 1, (100 * correct / total)))
            test_acc.append(100 * correct / total)

    np.savetxt('val_acc.csv', valid_acc, delimiter=',')
    np.savetxt('test_acc.csv', test_acc, delimiter=',')
    torch.save(net.state_dict(), 'fine_tune.pth')

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(test_loader, net, criterion, use_cuda)

if __name__ == "__main__":
    train_model()
