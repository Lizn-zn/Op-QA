import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable

from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader

from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import math

'''
Input selection and build gaussian model for c and r = c'/c
'''

import sys
sys.path.append('./data/')
import test_model as tm

sys.path.append('./model')
import resnet


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


np.random.seed(1)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'
use_cuda = torch.cuda.is_available()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence
        # padding of 2 is done below)
        self.BackBone = BackBone

    def forward(self, x):
        return self.BackBone(x)

    def hidden(self, x):
        for name, midlayer in self.BackBone._modules.items():
            if name != 'fc':
                x = midlayer(x)
            else:
                break
        x = torch.squeeze(x)
        return x

    def classifier(self, x):
        return self.BackBone.fc(x)


def orig_profit_curve(model, x_test, y_test, interval=0.05):
    # get confidence
    softmaxes = F.softmax(model.classifier(x_test.cuda()), dim=1)
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

    softmaxes = F.softmax(model.classifier(x_test.cuda()), dim=1)
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


def optimal_profit_curve(model, x_test, y_test, interval=0.05):
    softmaxes = F.softmax(model.classifier(x_test.cuda()), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
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
    softmaxes = F.softmax(model.classifier(x_op.cuda()), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
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
    test_batch = 128

    # load original model
    net = resnet.ResNet18()
    model_dir = './model/model_best.pth'
    # model_dir = './model/fine_tune.pth'
    state_dict = torch.load(model_dir)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    # net.load_state_dict(state_dict)
    net.eval()
    net.cuda()

    model = net

    # load operational data
    x_op, y_op = torch.load('./data/operational.pt')
    x_op = torch.FloatTensor(x_op)
    y_op = torch.LongTensor(y_op)
    test_op = TensorDataset(x_op, y_op)
    # test batch
    op_loader = torch.utils.data.DataLoader(
        test_op,
        batch_size=test_batch,
        shuffle=False,
    )

    # load operational test data
    x_test, y_test = torch.load('./data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    test_set = TensorDataset(x_test, y_test)
    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch,
        shuffle=False,
    )

    transform = transforms.Compose(
        [
            #        transforms.CenterCrop(32),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )
    dataset = tv.datasets.CIFAR10(
        './data/', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        dataset=dataset, batch_size=64, shuffle=True, num_workers=1)

    x_test = np.array([])
    y_test = np.array([])
    # load operational test data
    for inputs, targets in test_loader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_test = np.append(x_test, outputs.cpu().detach().numpy())
        y_test = np.append(y_test, targets.cpu().detach().numpy())
    x_test = torch.Tensor(x_test.reshape(-1, 512))
    y_test = torch.Tensor(y_test.reshape(-1))

    x_op = np.array([])
    y_op = np.array([])
    # load operational test data
    for inputs, targets in op_loader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(
            # inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_op = np.append(x_op, outputs.cpu().detach().numpy())
        y_op = np.append(y_op, targets.cpu().detach().numpy())
    x_op = torch.Tensor(x_op.reshape(-1, 512))
    y_op = torch.Tensor(y_op.reshape(-1))

    x_train = np.array([])
    y_train = np.array([])
    # load operational test data
    for inputs, targets in train_loader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(
            # inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_train = np.append(x_train, outputs.cpu().detach().numpy())
        y_train = np.append(y_train, targets.cpu().detach().numpy())
    x_train = torch.Tensor(x_train.reshape(-1, 512))
    y_train = torch.Tensor(y_train.reshape(-1))

    orig_profit_curve(model, x_test, y_test, interval=0.05)

    delta = evaluate(x_train, y_train, x_op, y_op, x_test, y_test)

    calibrated_profit_curve(model, delta, x_test, y_test)
