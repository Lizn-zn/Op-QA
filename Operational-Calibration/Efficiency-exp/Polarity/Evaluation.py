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

'''
Input selection and build gaussian model for c and r = c'/c
'''


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


np.random.seed(1)

device = torch.device("cpu")

select_size = 10

stat = 1
length_scale = 1

num_classes = 10


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # self.lstm = torch.nn.LSTM(256,64)
        # Fully connected layer
        # self.embedding = torch.nn.Embedding(vocab_size+1, 256)
        self.fc1 = torch.nn.Linear(2103, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 2)

    def forward(self, x):
        # x = self.lstm(x)
        # x = x[:, x.size(1)-1,:].squeeze(1)
        # x = x.view(x.size()[0],-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def hidden(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))

        return x


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
    softmaxes = F.softmax(model.fc3(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
    y_test = y_test.detach().numpy()

    deter = predictions == y_test
    index = np.where(confidences > lamda)[0]
    correct = np.sum(deter[index])
    incorrect = index.shape[0] - correct

    return correct, incorrect


def calibrated_profit(model, clf, x_test, y_test):
    softmaxes = F.softmax(model.fc3(x_test), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
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
        global M, C, cluster_label
        dist_mat = distance_matrix(fc_output)
        M, C = build_kmeans(dist_mat, init_num=init_size)
        temp = x_op.detach().numpy()
        from sklearn.metrics.pairwise import pairwise_distances
        dist = pairwise_distances(fc_output[M], temp, metric='euclidean')
        cluster_label = np.argmin(dist, axis=0)
        Flag = True
        while Flag:
            Flag = False
            for k in range(fc_output[M].shape[0]):
                temp_index = np.where(cluster_label == k)
                if temp_index[0].shape[0] == 0:
                    dist_mat = distance_matrix(fc_output)
                    M, C = build_kmeans(dist_mat, init_num=init_size)
                    dist = pairwise_distances(fc_output[M], temp, metric='euclidean')
                    cluster_label = np.argmin(dist, axis=0)
                    Flag = True

    global center_
    center_ = torch.Tensor(fc_output[M])
    center_ = center_.detach().numpy()

    x_select, y_select, select_index = input_initiation(
        x_op, y_op, M, C, init_size)

    softmaxes = F.softmax(model.fc3(x_op), dim=1)
    conf, predictions = torch.max(softmaxes, 1)
    conf = conf.detach().numpy()

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

        # _, index = input_selection(
            # x_op[no_select], conf[no_select], std[no_select], lamda, select_size, rand_select)
        _, index = input_selection(
            x_op, x_op[no_select], pred[no_select], std[no_select], lamda, select_size, rand_select)


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
    # load original model
    model_dir = './model/model.pth'
    model = Net()
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))

    # load operational test data
    x_test, y_test = torch.load('./data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    x_test, y_test = x_test.to(device), y_test.to(device)
    x_test = model.hidden(x_test)

    # load operational valid data
    x_valid, y_valid = torch.load('./data/operational.pt')
    x_valid = torch.FloatTensor(x_valid)
    y_valid = torch.LongTensor(y_valid)
    x_op, y_op = x_valid.to(device), y_valid.to(device)
    x_op = model.hidden(x_op)

    iters = 30
    init_size = 10
    select_size = 10

    statistic = 10

    # for lamda in [0.6, 0.7, 0.8, 0.9]:
    for lamda in [0.7,0.8,0.9]:
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
        plt.savefig(fold_dir + '/false.pdf', format='pdf', dpi=1000)
        plt.figure()
        plt.plot(x_axis, c1, 'b')
        plt.plot(x_axis, c2, 'r')
        plt.savefig(fold_dir + '/true.pdf', format='pdf', dpi=1000)

