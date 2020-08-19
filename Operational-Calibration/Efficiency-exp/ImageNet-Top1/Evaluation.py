import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable

from torch.nn import functional as F

import torchvision

import numpy as np
import matplotlib.pyplot as plt

from Input_initiation import input_initiation
from Input_selection import input_selection
from GP_build import conf_build, ratio_build, opc_predict

import warnings
warnings.filterwarnings('ignore')


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

    # def hidden(self, x):
    #     for name, midlayer in self.BackBone._modules.items():
    #         if name != 'fc':
    #             x = midlayer(x)
    #         else:
    #             break
    #     x = torch.squeeze(x)
    #     return x

    def hidden(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * \
            (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * \
            (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * \
            (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.BackBone.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.BackBone.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.BackBone.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.BackBone.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.BackBone.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.BackBone.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.BackBone.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.BackBone.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.BackBone.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.BackBone.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.BackBone.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.BackBone.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
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
    # load original model
    BackBone = torchvision.models.inception_v3(
        pretrained=True, transform_input=True)
    # BackBone = torchvision.models.resnet152(pretrained=True)
    net = Net()
    net.eval()
    net.cuda()

    x_test, y_test = torch.load('./data/test.pt')
    x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test).long()

    x_op, y_op = torch.load('./data/operational.pt')
    x_op, y_op = torch.Tensor(x_op), torch.Tensor(y_op).long()

    model = net

    iters = 100
    init_size = 30
    select_size = 10

    statistic = 10

    for lamda in [0.8, 0.9]:
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
