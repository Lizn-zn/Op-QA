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

import time
import os

import kernel_matrix

'''
Input selection and build gaussian model for c and r = c'/c
'''


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence
        # padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16 * 2 * 2, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    # last layer output

    def hidden(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))

        return x


def orig_profit(model, x_test, y_test):
    # get confidence
    softmaxes = F.softmax(model.fc3(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
    y_test = y_test.detach().numpy()

    index = np.where(predictions != y_test)
    corr = np.sum(confidences[index] > 0.9)
    print('high mis is ', corr)
    index = np.where(predictions == y_test)
    incorr = np.sum(confidences[index] > 0.9)
    print('high correct is ', incorr)

    deter = predictions == y_test
    score = np.square(deter - confidences)
    bs = np.mean(score)
    print('borel score is {}'.format(bs))
    acc = np.mean(deter)
    unc = acc * (1 - acc)
    print('uncertainy is {}'.format(unc))
    return corr, incorr, bs, unc


def calibrated_profit(model, clf, x_test, y_test, center_):
    softmaxes = F.softmax(model.fc3(x_test), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    y_test = y_test.detach().numpy()

    confidences, _ = opc_predict(model, clf, x_test, center_)
    indicate = predictions == y_test

    index = np.where(predictions != y_test)
    incorr = np.sum(confidences[index] > 0.9)
    # print('high mis is ', corr)
    index = np.where(predictions == y_test)
    corr = np.sum(confidences[index] > 0.9)
    # print('high correct is ', incorr)

    deter = predictions == y_test
    score = np.square(deter - confidences)
    bs = np.mean(score)
    # print('borel score is {}'.format(score))

    return corr, incorr, bs


def load_results(stat, init_size, final_size):
    orig_profit(model, x_test, y_test)

    corr, incorr, bs = 0, 0, 0
    for k in range(stat):
        ind = torch.load('exp_results/index-record/stat{0}.pt'.format(k))
        center_ind = ind[0:init_size]
        center_ = x_op[center_ind].detach().numpy()
        index = ind[0:final_size + init_size]
        x_select = x_op[index]
        y_select = y_op[index]
        select_index = index
        gp = ratio_build(model, x_op, x_select,
                         y_select, select_index, center_)

        c, i, b = calibrated_profit(model, gp, x_test, y_test, center_)
        corr += c
        incorr += i
        bs += b
    print('size is {0},high conf corr is {1}, high conf incorr is {2}, average bs is {3}'.format(
        final_size, corr / stat, incorr / stat, bs / stat))

if __name__ == '__main__':
    # load original model
    model_dir = 'model/LeNet-5.pth'
    model = Net()
    model.load_state_dict(torch.load(model_dir))

    # load operational test data
    x_test, y_test = torch.load('./data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    x_test = torch.unsqueeze(x_test, dim=1)
    x_test, y_test = x_test.to(device), y_test.to(device)
    x_test = model.hidden(x_test)

    # load operational valid data
    x_valid, y_valid = torch.load('./data/valid.pt')
    x_valid = torch.FloatTensor(x_valid)
    y_valid = torch.LongTensor(y_valid)
    x_valid = torch.unsqueeze(x_valid, dim=1)
    x_op, y_op = x_valid.to(device), y_valid.to(device)
    x_op = model.hidden(x_op)

    stat = 8
    init_size = 10
    final_size = 600
    load_results(stat, init_size, final_size)
