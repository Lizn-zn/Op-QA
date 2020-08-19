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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence
        # padding of 2 is done below)
        self.BackBone = BackBone

    def forward(self, x):
        return self.BackBone(x)

    # def hidden(self, x):
    #     for name, midlayer in self.BackBone._modules.items():
    #         if name != 'fc':
    #             x = midlayer(x)
    #         else:
    #             break
    #     x = torch.squeeze(x)
    #     return x

    def hidden(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * \
            (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * \
            (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * \
            (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.BackBone.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.BackBone.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.BackBone.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.BackBone.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.BackBone.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.BackBone.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.BackBone.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.BackBone.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.BackBone.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.BackBone.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.BackBone.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        return x

    def classifier(self, x):
        return self.fc(x)


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

    global BackBone
    BackBone = tv.models.inception_v3(
        pretrained=True, transform_input=True)
    # BackBone = torchvision.models.resnet152(pretrained=True)
    net = Net()
    net.cuda()

    optimizer = optim.SGD(net.BackBone.parameters(), lr=1e-2, momentum=0.9)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    net.train()

    valid_acc = []
    test_acc = []

    for i, data in enumerate(data_loader):
        inputs, labels = data
        if i >= 0 and i < 5:
            train_set = TensorDataset(inputs, labels)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=train_batch,
                shuffle=True,
            )
        elif i >= 5 and i < 10:
            test_set = TensorDataset(inputs, labels)
            # test batch
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=test_batch,
                shuffle=False,
            )
        else:
            break

        for epoch in range(EPOCH):
            for i, data in enumerate(train_loader):
                inputs, labels = (inputs).cuda(), labels.cuda()
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

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.405],
                                     std=[0.229, 0.224, 0.225])
    transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomAffine(0, (0.2, 0.2)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    imagenet_data = tv.datasets.ImageNet(
        'val-data/', split='val', download=True, transform=transforms)
    data_loader = torch.utils.data.DataLoader(imagenet_data,
                                              batch_size=1000, shuffle=False,
                                              num_workers=4)

    train_model()
