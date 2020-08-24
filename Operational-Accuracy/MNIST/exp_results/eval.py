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

device = torch.device("cpu")

warnings.filterwarnings('ignore')


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


def compute_accuracy(model, x_test, y_test):
    # get confidence
    softmaxes = F.softmax(model.fc3(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
    y_test = y_test.detach().numpy()

    deter = predictions == y_test
    return np.mean(deter)


if __name__ == '__main__':
    # load original model
    model_dir = '../model/LeNet-5.pth'
    model = Net()
    model.load_state_dict(torch.load(model_dir))

    # load operational test data
    x_test, y_test = torch.load('../data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    x_test = torch.unsqueeze(x_test, dim=1)
    x_test, y_test = x_test.to(device), y_test.to(device)
    x_test = model.hidden(x_test)

    accuracy = compute_accuracy(model, x_test, y_test)
    print('The actual accuracy of model is {}'.format(accuracy))

    rand = np.loadtxt('random_accuracy.csv', delimiter=',')
    select = np.loadtxt('select_accuracy.csv', delimiter=',')

    rand_var = np.mean(np.square(rand - accuracy), axis=0)
    select_var = np.mean(np.square(select - accuracy), axis=0)


    print('relative effective is {}'.format(np.mean(select_var/rand_var)))

    x_axis = [i for i in range(1, rand_var.shape[0]+1)]
    x_axis = np.array(x_axis) + 30
    y_axis1 = rand_var
    y_axis2 = select_var
    plt.figure()
    plt.plot(x_axis, y_axis1, 'b')
    plt.plot(x_axis, y_axis2, 'r')
    plt.savefig('result.pdf', format='pdf', dpi=1000)
