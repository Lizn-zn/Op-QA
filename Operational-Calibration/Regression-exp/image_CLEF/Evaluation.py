
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


'''
Input selection and build gaussian model for c and r = c'/c
'''

import Data_load as dl

import sys
sys.path.append('./data/')
import data_process as dp

sys.path.append('./model/')
import test_model as tm


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7,8,9'


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


np.random.seed(1)

use_cuda = torch.cuda.is_available()

select_size = 300

stat = 1
length_scale = 1

num_classes = 12


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


def distance_matrix(fc_output):
    # compute the distance matrix on representation space
    X = fc_output
    # m, n = X.shape
    # G = np.dot(X, X.T)
    # H = np.tile(np.diag(G), (m, 1))
    # return H + H.T - 2 * G
    from sklearn.metrics.pairwise import pairwise_distances
    return pairwise_distances(X, metric='euclidean')


def build_kmeans(dist_mat, init_num=50):
    # use kMedoids to build cluster
    from sklearn.cluster import KMeans
    from kMedoids import kMedoids
    M, C = kMedoids(dist_mat, init_num)
    return M, C


def pred_devide(pred):
    # return pred
    pred = np.around(pred, decimals=1)
    return pred


def orig_profit_curve(model, x_test, y_test, interval=0.05):
    x_test = x_test.cuda()
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
    print('uncertainy is {}'.format(acc * (1-acc)))


def calibrated_profit_curve(model, clf, x_test, y_test, interval=0.05):
    x_test = x_test.cuda()
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()

    confidences, _ = opc_predict(model, clf, x_test, center_)

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
    x_test = x_test.cuda()
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


def evaluate(x_op, y_op, init_size=30, iteration=18):
    '''
    evaluate the model for profit
    :param x_op: operational data in representation space
    :param y_op: operational label
    :param init_size: size for initial examples
    :param iteration: iterative size
    : return 
    '''

    # build distance matrix
    fc_output = x_op.detach().numpy()
    dist_mat = distance_matrix(fc_output)
    M, C = build_kmeans(dist_mat, init_num=init_size)
    global center_
    center_ = torch.Tensor(fc_output[M])
    center_ = center_.detach().numpy()

    x_select, y_select, select_index = input_initiation(
        x_op, y_op, M, C, init_size)

    # gp1 = conf_build(net, x_op, center_)

    for i in range(iteration):
        print("sample size: {}".format((i) * select_size + init_size))
        index = np.ones((x_op.shape[0],))
        index[select_index] = 0
        no_select = np.where(index == 1)[0]

        gp = ratio_build(net, x_op, x_select,
                         y_select, select_index, center_)

        pred, _ = opc_predict(net, gp, x_op, center_)

        # select
        _, index = input_selection(
            x_op[no_select], select_size, rand_select=True)
        temp = no_select[index]
        select_index = np.append(select_index, temp)
        x_select = x_op[select_index]
        y_select = y_op[select_index]

    return gp


if __name__ == '__main__':
    # load original model
    model_dir = './model/best_resnet_c.pth'
    net = Net()
    net.BackBone = tv.models.resnet50(pretrained=False)
    net.BackBone.fc = nn.Linear(2048, 12)

    net.BackBone.load_state_dict(torch.load(
        model_dir, map_location=torch.device('cpu')))
    net.cuda()

    op_loader, test_loader = dl.load_imageclef_train(
        './data/', 'p', 24, 'src')

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = tm.test(
        test_loader, net.BackBone, criterion, use_cuda)

    x_test = np.array([])
    y_test = np.array([])
    # load operational test data
    for inputs, targets in test_loader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(
            # inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_test = np.append(x_test, outputs.cpu().detach().numpy())
        y_test = np.append(y_test, targets.cpu().detach().numpy())
    x_test = torch.Tensor(x_test.reshape(-1, 2048))
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
    x_op = torch.Tensor(x_op.reshape(-1, 2048))
    y_op = torch.Tensor(y_op.reshape(-1))

    iters = 3
    init_size = 1
 
    fig = plt.figure()

    orig_profit_curve(net, x_test, y_test)

    gp = evaluate(x_op, y_op, init_size=init_size, iteration=iters)

    calibrated_profit_curve(net, gp, x_test, y_test, interval=0.05)

    optimal_profit_curve(net, x_test, y_test)

    plt.savefig('curve.eps', format='eps', dpi=1000)
