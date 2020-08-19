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

import math

'''
Input selection and build gaussian model for c and r = c'/c
'''

import Data_load as dl

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
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7, 8, 9'
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


def orig_profit(model, x_test, y_test):
    # get confidence
    softmaxes = F.softmax(model.classifier(x_test.cuda()), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    y_test = y_test.detach().numpy()

    deter = predictions == y_test
    index = np.where(confidences > lamda)[0]
    correct = np.sum(deter[index])
    incorrect = index.shape[0] - correct

    return correct, incorrect


def calibrated_profit(model, clf, x_test, y_test):
    softmaxes = F.softmax(model.classifier(x_test.cuda()), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    y_test = y_test.detach().numpy()

    confidences, _ = opc_predict(model, clf, x_test, center_)

    deter = predictions == y_test
    index = np.where(confidences > lamda)[0]
    correct = np.sum(deter[index])
    incorrect = index.shape[0] - correct

    return correct, incorrect


def evaluate(x_op, y_op, x_test, y_test, init_size=30, iteration=18, rand_select=True):
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
    if rand_select == True:
        global M, C
        dist_mat = distance_matrix(fc_output)
        M, C = build_kmeans(dist_mat, init_num=init_size)

    global center_
    center_ = torch.Tensor(fc_output[M])
    center_ = center_.detach().numpy()

    x_select, y_select, select_index = input_initiation(
        x_op, y_op, M, C, init_size)

    softmaxes = F.softmax(model.classifier(x_op.cuda()), dim=1)
    conf, predictions = torch.max(softmaxes, 1)
    conf = conf.cpu().detach().numpy()

    corr_list = []
    incorr_list = []

    for i in range(iteration):
        # print("sample size: {}".format((i) * select_size + init_size))
        index = np.ones((x_op.shape[0],))
        index[select_index] = 0
        no_select = np.where(index == 1)[0]

        gp = ratio_build(model, x_op, x_select,
                         y_select, select_index, center_)

        pred, std = opc_predict(model, gp, x_op, center_)

        _, index = input_selection(
            x_op, x_op[no_select], pred[no_select], std[no_select], lamda, select_size, rand_select)
        # print(conf[no_select[index]])
        temp = no_select[index]
        # save selected index
        select_index = np.append(select_index, temp)
        x_select = x_op[select_index]
        y_select = y_op[select_index]

        correct, incorrect = calibrated_profit(model, gp, x_test, y_test)
        corr_list.append(correct)
        incorr_list.append(incorrect)

        print('iteration {0}, lamda is {1}, high conf mis is {2}, correct is {3}'.format(
            i, lamda, incorrect, correct))
    return gp, corr_list, incorr_list

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
    
    x_temp, y_temp = torch.load('./data/original.pt')
    x_temp = torch.FloatTensor(x_temp)
    y_temp = torch.LongTensor(y_temp)
    temp_set = TensorDataset(x_temp, y_temp)
    # test batch
    temp_loader = torch.utils.data.DataLoader(
        temp_set,
        batch_size=test_batch,
        shuffle=False,
    ) 
   
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = tm.test(
        temp_loader, net, criterion, use_cuda)

    test_loss, test_acc = tm.test(
        test_loader, net, criterion, use_cuda)

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



    iters = 100
    init_size = 30
    select_size = 10

    statistic = 10

    for lamda in [0.9]:
        correct, incorrect = orig_profit(model, x_test, y_test)
        print('original, lamda is {0}, high conf mis is {1}, correct is {2}'.format(
            lamda, incorrect, correct))

        fold_dir = './exp_results/' + str(lamda)
        c1 = np.zeros((iters))
        c2 = np.zeros((iters))
        ic1 = np.zeros((iters))
        ic2 = np.zeros((iters))
        for k in range(statistic):
            print('for statistic, repect the {}th time'.format(k))
            _,  tmp_c1, tmp_ic1 = evaluate(x_op, y_op, x_test, y_test,
                                           init_size=init_size, iteration=iters, rand_select=True)
            _,  tmp_c2, tmp_ic2 = evaluate(x_op, y_op, x_test, y_test,
                                           init_size=init_size, iteration=iters, rand_select=False)

            c1 = c1 + np.array(tmp_c1)
            c2 = c2 + np.array(tmp_c2)
            ic1 = ic1 + np.array(tmp_ic1)
            ic2 = ic2 + np.array(tmp_ic2)
       
        import os 
        if os.path.exists(fold_dir) == False:
            os.makedirs(fold_dir)          

        np.savetxt(fold_dir + '/random_c.csv', c1)
        np.savetxt(fold_dir + '/select_c.csv', c2)
        np.savetxt(fold_dir + '/random_ic.csv', ic1)
        np.savetxt(fold_dir + '/select_ic.csv', ic2)

        x_axis = [i for i in range(1, iters + 1)]
        y_axis1 = ic1
        y_axis2 = ic2
        plt.figure()
        plt.plot(x_axis, y_axis1, 'b')
        plt.plot(x_axis, y_axis2, 'r')
        plt.savefig(fold_dir + '/false.eps', format='pdf', dpi=1000)
        plt.figure()
        plt.plot(x_axis, c1, 'b')
        plt.plot(x_axis, c2, 'r')
        plt.savefig(fold_dir + '/true.eps', format='pdf', dpi=1000)

