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


class VGG(nn.Module):

    def __init__(self, features, num_classes=100):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def hidden(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}





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
    test_batch = 32

    x_train, y_train = torch.load('../data/operational.pt')
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    train_set = TensorDataset(x_train, y_train)

    # train batch
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch,
        shuffle=True,
    )

    x_test, y_test = torch.load('../data/testing.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    test_set = TensorDataset(x_test, y_test)
    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch,
        shuffle=False,
    )



    # load original model
    net = VGG(make_layers(cfg['E'], batch_norm=True))
    # net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load('./vgg19.pth.tar')
    state_dict = checkpoint['state_dict']
    # net.load_state_dict(state_dict)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'classifier' in k:
            # continue
            # name = 'module.' + k
            name = k
        else:
            #         name = k.split('.')
            #         name[1], name[0] = name[0], name[1]
            #         name = '.'.join(name)
            name = k.replace('module.', '')
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.cuda()


    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(test_loader, net, criterion, use_cuda)

    criterion = nn.CrossEntropyLoss()
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
            valid_acc.append(correct / float(total))

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
            test_acc.append(correct / float(total))

    np.savetxt('val_acc.csv', valid_acc, delimiter=',')
    np.savetxt('test_acc.csv', test_acc, delimiter=',')
    torch.save(net.state_dict(), 'fine_tune.pth')

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = test(test_loader, net, criterion, use_cuda)

if __name__ == "__main__":
    train_model()
