import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable

from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

from Input_initiation import input_initiation
from Input_selection import input_selection
from GP_build import conf_build, ratio_build, opc_predict

import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)

import math


import sys
sys.path.append('./data/')
import test_model as tm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7, 8, 9'
use_cuda = torch.cuda.is_available()

'''
Input selection and build gaussian model for c and r = c'/c
'''


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
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


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


def baseline_profit_curve(model, x_test, y_test, interval=0.05):
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()

    print(x_test.shape[0])
    print(predictions)
    print(y_test)
    print(np.sum(predictions == y_test.cpu().detach().numpy()))

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test.cpu().detach().numpy())
        indicate2 = (predictions != y_test.cpu().detach().numpy())
        ell1 = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        ell2 = indicate1 * (1 - lamda)
        ell1 = np.sum(ell1) / x_test.shape[0]
        ell2 = np.sum(ell2) / x_test.shape[0]
        y_axis.append(ell2 - ell1)
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'r')
    print(y_axis)
    np.savetxt('exp_results/baseline.csv', y_axis)


if __name__ == '__main__':

    # load original model
    net = VGG(make_layers(cfg['E'], batch_norm=True))
    # net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load('./model/model_best.pth.tar')
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
    net.eval()

    # load operational test data
    x_test, y_test = torch.load('./data/training.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    x_test, y_test = x_test.cuda(), y_test.cuda()
    dataset = TensorDataset(x_test, y_test)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=28,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = tm.test(
        data_loader, net, criterion, use_cuda)

    fc_output = np.array([])
    for inputs, targets in data_loader:
        temp_output = net.hidden(inputs)
        _, pred = torch.max(net.classifier(temp_output), 1)
        print(pred)
        print(targets)
        temp = temp_output.cpu().detach().numpy().reshape(-1, 512)
        fc_output = np.append(
            fc_output, temp)
        fc_output = fc_output.reshape(-1, 512)

    print(y_test[0:3])
    fc_output = fc_output.reshape(-1, 512)
    x_test = torch.Tensor(fc_output).cuda()
    _, pred = torch.max(net.classifier(torch.Tensor(fc_output[0:3]).cuda()), 1)
    print(pred)

    fig = plt.figure()

    baseline_profit_curve(net, x_test, y_test, interval=0.05)

    plt.show()
