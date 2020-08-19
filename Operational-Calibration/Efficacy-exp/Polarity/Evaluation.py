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


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


np.random.seed(1)

device = torch.device("cpu")

select_size = 497



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

    def classifier(self, x):
        x = self.fc3(x)
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
    print('uncertainy is {}'.format(acc * (1-acc)))


def calibrated_profit_curve(model, clf, x_test, y_test, interval=0.05):
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

    # gp1 = conf_build(model, x_op, center_)

    for i in range(iteration):
        print("sample size: {}".format((i) * select_size + init_size))
        index = np.ones((x_op.shape[0],))
        index[select_index] = 0
        no_select = np.where(index == 1)[0]

        gp = ratio_build(model, x_op, x_select,
                         y_select, select_index, center_)

        pred, _ = opc_predict(model, gp, x_op, center_)

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
    model_dir = './model/model.pth'
    model = Net()
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    mdoel = model.cpu()
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

    iters = 3
    init_size = 10

    fig = plt.figure()

    orig_profit_curve(model, x_test, y_test)

    optimal_profit_curve(model,  x_test, y_test, interval=0.05)

    gp = evaluate(x_op, y_op, init_size=init_size, iteration=iters)

    calibrated_profit_curve(model, gp, x_test, y_test, interval=0.05)

    plt.savefig('curve.eps', format='eps', dpi=1000)
