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


import Data_load as dl

import sys
sys.path.append('./data/')
import data_process as dp

sys.path.append('./model/')
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence
        # padding of 2 is done below)
        BackBone = tv.models.resnet50(pretrained=False)
        BackBone.fc = nn.Linear(2048, 12)
        self.BackBone = BackBone

    def forward(self, x):
        return self.BackBone(x)

    def hidden(self, x):
        for name, midlayer in self.BackBone._modules.items():
            if name != 'fc':
                x = midlayer(x)
            else:
                break
        return x

    def classifier(self, x):
        return self.BackBone.fc(x)


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

    model_dir = './model/best_resnet_c.pth'
    net = Net()
    net.BackBone = tv.models.resnet50(pretrained=False)
    net.BackBone.fc = nn.Linear(2048, 12)

    net.BackBone.load_state_dict(torch.load(
        model_dir, map_location=torch.device('cpu')))
    net.cuda()

    op_loader, test_loader = dl.load_imageclef_train(
        './data/', 'c', 48, 'tar')

    data_loader = test_loader

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = tm.test(
        data_loader, net, criterion, use_cuda)

    fc_output = np.array([])
    y_test = np.array([])
    for inputs, targets in data_loader:
        temp_output = net.hidden(inputs.cuda())
        temp_output = torch.squeeze(temp_output)
        _, pred = torch.max(net.classifier(temp_output), 1)
        print(pred)
        print(targets)
        temp = temp_output.cpu().detach().numpy().reshape(-1, 2048)
        fc_output = np.append(
            fc_output, temp)
        fc_output = fc_output.reshape(-1, 2048)
        y_test = np.append(y_test, targets.cpu().detach().numpy())
    y_test = y_test.reshape(-1)
    y_test = torch.Tensor(y_test)
    print(y_test[0:3])
    fc_output = fc_output.reshape(-1, 2048)
    x_test = torch.Tensor(fc_output).cuda()
    _, pred = torch.max(net.classifier(torch.Tensor(fc_output[0:3]).cuda()), 1)
    print(pred)

    fig = plt.figure()

    baseline_profit_curve(net, x_test, y_test, interval=0.05)

    plt.show()
