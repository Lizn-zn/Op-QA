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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7, 8, 9'
use_cuda = torch.cuda.is_available()
import math

'''
Input selection and build gaussian model for c and r = c'/c
'''


import Data_load as dl

import sys
sys.path.append('./data/')
import data_process as dp

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


np.random.seed(1)

device = torch.device("cuda")

select_size = 445

stat = 1
length_scale = 1


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


def bs_score(conf, y_pred, y_truth, accuracy):
    fk = np.mean(conf)
    ok = np.mean((y_pred == y_truth))
    nk = conf.shape[0]

    o = accuracy

    reliability = nk * np.square(fk - ok)
    resolution = nk * np.square(ok - o)
    return reliability, resolution


def orig_profit_curve(model, x_test, y_test, interval=0.05):
    # get confidence
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()

    index = np.where(predictions != y_test)
    print("high conf misclassified are {}".format(
        np.sum(confidences[index] > 0.9)))
    index = np.where(predictions == y_test)
    print("high conf correct are {}".format(np.sum(confidences[index] > 0.9)))

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test)
        indicate2 = (predictions != y_test)
        ell = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'b')
    np.savetxt('exp_results/original.csv', y_axis)

    accuracy = np.mean(predictions == y_test)

    bin_boundaries = np.linspace(0, 1, 50 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    reliability, resolution = 0, 0
    for k in range(50):
        ind = np.where(np.logical_and(
            confidences >= bin_lowers[k], confidences < bin_uppers[k]))
        if ind[0].shape[0] == 0:
            continue
        r1, r2 = bs_score(confidences[ind],
                          predictions[ind], y_test[ind], accuracy)
        reliability += r1
        resolution += r2
    print("reliability is {0}, resolution is {1}".format(
        reliability / x_test.shape[0], resolution / x_test.shape[0]))
    deter = predictions == y_test
    print('brier score is {0}'.format(
        np.mean(np.square(deter - confidences))))
    print('accuracy is {}'.format(np.sum(deter)))


def calibrated_profit_curve(model, test_loader, y_test, interval=0.05):
    y_test = y_test.cpu().detach().numpy()

    confidences = np.array([])
    predictions = np.array([])
    # load operational test data
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        softmaxes = F.softmax(model(inputs), dim=1)
        conf, pred = torch.max(softmaxes, 1)
        confidences = np.append(confidences, conf.cpu().detach().numpy())
        predictions = np.append(predictions, pred.cpu().detach().numpy())

    index = np.where(predictions != y_test)
    print("high conf misclassified are {}".format(
        np.sum(confidences[index] > 0.9)))
    index = np.where(predictions == y_test)
    print("high conf correct are {}".format(np.sum(confidences[index] > 0.9)))

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test)
        indicate2 = (predictions != y_test)
        ell = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'r')
    np.savetxt('exp_results/calibrate.csv', y_axis)

    accuracy = np.mean(predictions == y_test)
    bin_boundaries = np.linspace(0, 1, 50 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    reliability, resolution = 0, 0
    for k in range(50):
        ind = np.where(np.logical_and(
            confidences >= bin_lowers[k], confidences < bin_uppers[k]))
        if ind[0].shape[0] == 0:
            continue
        r1, r2 = bs_score(confidences[ind],
                          predictions[ind], y_test[ind], accuracy)
        reliability += r1
        resolution += r2
    print("reliability is {0}, resolution is {1}".format(
        reliability / x_test.shape[0], resolution / x_test.shape[0]))

    deter = predictions == y_test
    print('brier score is {0}'.format(
        np.mean(np.square(deter - confidences))))


def optimal_profit_curve(model,  x_test, y_test, interval=0.05):
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    indicate = (predictions == y_test.cpu().detach().numpy())

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        ell = indicate * (1 - lamda)
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'k')
    np.savetxt('exp_results/optimal.csv', y_axis)


if __name__ == '__main__':

    # load original model
    model_dir = './model/best_resnet_c.pth'
    net = Net()
    net.BackBone = tv.models.resnet50(pretrained=False)
    net.BackBone.fc = nn.Linear(2048, 12)

    net.BackBone.load_state_dict(torch.load(
        model_dir, map_location=torch.device('cpu')))
    net.eval()
    net.cuda()

    op_loader, test_loader = dl.load_imageclef_train(
        './data/', 'p', 24, 'src')

    x_test = np.array([])
    y_test = np.array([])
    # load operational test data
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_test = np.append(x_test, outputs.cpu().detach().numpy())
        y_test = np.append(y_test, targets.cpu().detach().numpy())
    x_test = torch.Tensor(x_test.reshape(-1, 2048)).cuda()
    y_test = torch.Tensor(y_test.reshape(-1)).cuda()

    valid_loader = op_loader

    iters = 3
    init_size = 30

    fig = plt.figure()

    orig_profit_curve(net, x_test, y_test)

    optimal_profit_curve(net,  x_test, y_test, interval=0.05)

    from temperature_scaling import ModelWithTemperature
    scaled_model = ModelWithTemperature(net)
    scaled_model.set_temperature(valid_loader)
    scaled_model.cuda()

    calibrated_profit_curve(scaled_model, test_loader, y_test, interval=0.05)

    plt.savefig('curve.eps', format='eps', dpi=1000)
