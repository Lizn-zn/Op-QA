import warnings
import os
import time
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

import math


import sys
sys.path.append('../data/')
import test_model as tm


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7, 8, 9'
use_cuda = torch.cuda.is_available()

warnings.filterwarnings('ignore')

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
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]    


def compute_accuracy(model, x_test, y_test):
    # get confidence
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
    y_test = y_test.detach().numpy()

    deter = predictions == y_test
    return np.mean(deter)


if __name__ == '__main__':
    test_batch = 5

    # load original model
    net = VGG(make_layers(cfg['E'], batch_norm=True))
    # net = torch.nn.DataParallel(net).cuda()
    checkpoint = torch.load('../model/vgg19.pth.tar')
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
   
    # load operational test data
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

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = tm.test(
        test_loader, net, criterion, use_cuda)

    accuracy = test_acc / 100
    print('The actual accuracy of model is {}'.format(accuracy))

    rand = np.loadtxt('random_accuracy.csv', delimiter=',')
    select = np.loadtxt('select_accuracy.csv', delimiter=',')

    rand_var = np.mean(np.square(rand - accuracy), axis=0)
    select_var = np.mean(np.square(select - accuracy), axis=0)

    print('relative effective is {}'.format(np.mean(select_var/rand_var)))

    x_axis = [i for i in range(1, rand_var.shape[0]+1)]
    y_axis1 = rand_var
    y_axis2 = select_var
    plt.figure()
    plt.plot(x_axis, y_axis1, 'b')
    plt.plot(x_axis, y_axis2, 'r')
    plt.savefig('result.pdf', format='pdf', dpi=1000)
