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

from Input_select import input_select

import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans


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
