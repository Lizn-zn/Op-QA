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

import warnings
warnings.filterwarnings('ignore')


from sklearn.cluster import KMeans

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

num_classes = 2


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


def compute_accuracy(model, x, y):
    # get confidence
    softmaxes = F.softmax(model.fc3(x), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
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

    num_cluster = np.minimum(num_classes, 2)
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
