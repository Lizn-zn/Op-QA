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

from Input_select import input_select

from sklearn.cluster import KMeans

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

num_classes = 10

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


def compute_accuracy(model, x, y):
    # get confidence
    softmaxes = F.softmax(model.classifier(x.cuda()), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    y = y.detach().numpy()
    deter = predictions == y
    return np.mean(deter), np.var(deter)

def weighted_accuracy(model, cluster, ws, x, y):
    label = cluster.predict(x.cpu().detach().numpy())
    # num = x.shape[0]
    accuracies = []
    variances = []
    acc = 0
    w = 0
    for i in range(np.max(label)+1):
        ind = np.where(label == i)[0]
        if ind.shape[0] == 0:
            accuracies.append(0)
            variances.append(1e6)
            w += ws[i]
        else:
            tmp_acc, tmp_var = compute_accuracy(model, x[ind], y[ind])
            accuracies.append(tmp_acc)
            variances.append(tmp_var)
            acc += tmp_acc * ws[i]
    acc = acc / (1-w)
    return acc, variances 

def get_weights(cluster, x):
    ws = []
    label = cluster.predict(x.cpu().detach().numpy())
    num = x.shape[0]
    for i in range(np.max(label)+1):
        ind = np.where(label == i)[0]    
        ws.append(ind.shape[0]/num)
    ws = np.array(ws)
    return ws

def convert2score(ws):
    ws = np.array(ws)
    ws = ws / np.sum(ws)
    scores = [0]
    for i in range(ws.shape[0]):
        tmp = scores[i] + ws[i]
        scores.append(tmp)
    scores = np.array(scores) 
    return scores 

def evaluate(x_op, y_op, sample_size, statistic=50):
    '''
    evaluate the model for profit
    :param x_op: operational data in representation space
    :param y_op: operational label
    '''

    num_cluster = np.minimum(num_classes, 6)
    print('cluster num: {}'.format(num_cluster))
    from sklearn.cluster import KMeans
    cluster = KMeans(n_clusters=num_cluster).fit(x_op.cpu().detach().numpy())

    ws = get_weights(cluster, x_op)

    random_acc = []
    select_acc = []

    for i in range(statistic):
        rand_index = input_select(cluster, x_op, ws, sample_size, rand_select=True)
        x_rand = x_op[rand_index]
        y_rand = y_op[rand_index]

        ce_index = input_select(cluster, x_op, ws, sample_size, rand_select=False)
        x_select = x_op[ce_index]
        y_select = y_op[ce_index]


        r_acc, _ = compute_accuracy(model, x_rand, y_rand)
        s_acc, _ = weighted_accuracy(model, cluster, ws, x_select, y_select)
        select_acc.append(s_acc)
        random_acc.append(r_acc)

    select_acc = np.array(select_acc)
    random_acc = np.array(random_acc)

    return select_acc, random_acc

    
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

    statistic = 500
    iters = 30
    init_size = 30
    inc_size = 5

    accuracy, _ = compute_accuracy(model, x_test, y_test)
    print('The actual accuracy of model is {}'.format(accuracy))

    fold_dir = './exp_results/' 
    select_accuracy = np.zeros((statistic, iters))
    random_accuracy = np.zeros((statistic, iters))

    for k in range(iters):
        sample_size = init_size + inc_size * k
        print('size: {}'.format(sample_size))
        select_acc, random_acc = evaluate(x_op, y_op, sample_size, statistic=statistic)
        print(np.mean(select_acc), np.mean(random_acc))
        print('re: {}'.format(np.mean(np.var(select_acc)/np.var(random_acc))))

        select_accuracy[:,k] = np.array(select_acc)
        random_accuracy[:, k] =  np.array(random_acc)

    if os.path.exists(fold_dir) == False:
        os.makedirs(fold_dir)

    np.savetxt(fold_dir + '/select_accuracy.csv', select_accuracy, delimiter=',')
    np.savetxt(fold_dir + '/random_accuracy.csv', random_accuracy, delimiter=',')

