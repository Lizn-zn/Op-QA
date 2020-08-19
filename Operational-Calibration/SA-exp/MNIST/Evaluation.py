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

import warnings
warnings.filterwarnings('ignore')

import time
import os


'''
Input selection and build gaussian model for c and r = c'/c
'''


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


np.random.seed(1)

device = torch.device("cpu")


stat = 1
length_scale = 1

num_classes = 10


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

    def classifier(self, x):
        x = self.fc3(x)
        return x


def orig_profit_curve(model, x_test, y_test, interval=0.05):
    # get confidence
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()

    index = np.where(predictions != y_test.detach().numpy())
    print('high mis is ', np.sum(confidences[index] > 0.9))
    index = np.where(predictions == y_test.detach().numpy())
    print('high correct is ', np.sum(confidences[index] > 0.9))
    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test.detach().numpy())
        indicate2 = (predictions != y_test.detach().numpy())
        ell = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'b')
    np.savetxt('exp_results/original.csv', y_axis)
    deter = predictions == y_test.detach().numpy()
    score = np.square(deter - confidences)
    print('borel score is {}'.format(np.mean(score)))
    acc = np.mean(deter)
    print('uncertainy is {}'.format(acc * (1 - acc)))


def calibrated_profit_curve(model, delta, x_test, y_test, interval=0.05):

    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()

    confidences = confidences.cpu().detach().numpy() + delta
    confidences = np.clip(confidences, 0, 1)

    # index = np.where(predictions == y_test.detach().numpy())
    # print(confidences[index])

    index = np.where(predictions != y_test.detach().numpy())
    print('high mis is ', np.sum(confidences[index] > 0.9))
    index = np.where(predictions == y_test.detach().numpy())
    print('high correct is ', np.sum(confidences[index] > 0.9))

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test.detach().numpy())
        indicate2 = (predictions != y_test.detach().numpy())
        ell = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'r')
    np.savetxt('exp_results/calibrate.csv', y_axis)
    deter = predictions == y_test.detach().numpy()
    score = np.square(deter - confidences)
    print('borel score is {}'.format(np.mean(score)))


def optimal_profit_curve(model,  x_test, y_test, interval=0.05):
    softmaxes = F.softmax(model.fc3(x_test), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    indicate = (predictions == y_test.detach().numpy())

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        ell = indicate * (1 - lamda)
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'k')
    np.savetxt('exp_results/optimal.csv', y_axis)


def evaluate(x_train, y_train, x_op, y_op, x_test, y_test):
    '''
    evaluate the model for profit
    :param x_op: operational data in representation space
    :param y_op: operational label
    :param init_size: size for initial examples
    :param iteration: iterative size
    : return 
    '''

    import SA
    lsa = SA.fetch_lsa(model, x_train, x_op)

    softmaxes = F.softmax(model.fc3(x_op), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
    confidences = np.clip(confidences, 0, 1)
    temp = np.clip((predictions == y_op.detach().numpy()), 0, 1)
    # regress = np.log(temp) - np.log(confidences)
    regress = temp - confidences

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    poly = PolynomialFeatures(degree=10)
    X = poly.fit_transform(np.array(lsa).reshape(-1, 1))
    print(poly)
    lr = LinearRegression()
    lr.fit(X, y_op.cpu().detach().numpy())
    print(lr)

    lsa = SA.fetch_lsa(model, x_train, x_test)
    X = poly.transform(np.array(lsa).reshape(-1, 1))
    delta = lr.predict(X)
    return delta

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

    # load operational valid data
    x_train, y_train = torch.load('./data/training.pt')
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_train = torch.unsqueeze(x_train, dim=1)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_train = model.hidden(x_train)

    orig_profit_curve(model, x_test, y_test, interval=0.05)

    delta = evaluate(x_train, y_train, x_op, y_op, x_test, y_test)

    calibrated_profit_curve(model, delta, x_test, y_test)
