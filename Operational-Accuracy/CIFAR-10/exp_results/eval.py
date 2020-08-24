import warnings
import os
import time
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


import sys
sys.path.append('../data/')
import test_model as tm

sys.path.append('../model')
import resnet

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7, 8, 9'
use_cuda = torch.cuda.is_available()

warnings.filterwarnings('ignore')


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


def compute_accuracy(model, x_test, y_test):
    # get confidence
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
    y_test = y_test.detach().numpy()

    deter = predictions == y_test
    return np.mean(deter)


if __name__ == '__main__':
    test_batch = 128

    # load original model
    net = resnet.ResNet18()
    model_dir = '../model/model_best.pth'
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

    # load operational test data
    x_test, y_test = torch.load('../data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    test_set = TensorDataset(x_test, y_test)
    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch,
        shuffle=False,
    )
    
   
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc = tm.test(
        test_loader, net, criterion, use_cuda)

    accuracy = test_acc / 100
    print('The actual accuracy of model is {}'.format(accuracy))

    rand = np.loadtxt('random_accuracy.csv', delimiter=',')
    select = np.loadtxt('select_accuracy.csv', delimiter=',')

    rand_var = np.mean(np.square(rand - accuracy), axis=0)
    select_var = np.mean(np.square(select - accuracy), axis=0)

    print('relative effective is {}'.format(np.mean(select_var/rand_var)))

    x_axis = [i for i in range(1, rand_var.shape[0]+1)]
    y_axis1 = rand_var
    y_axis2 = select_var
    plt.figure()
    plt.plot(x_axis, y_axis1, 'b')
    plt.plot(x_axis, y_axis2, 'r')
    plt.savefig('result.pdf', format='pdf', dpi=1000)
