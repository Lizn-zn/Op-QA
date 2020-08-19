import pandas as pd
import numpy as np
import torch

from torch.nn import functional as F

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from tensorflow.contrib import learn
import tensorflow as tf
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def test_model(model):
    x_test, y_test = torch.load('../data/tar.pt')
    x_test = torch.Tensor(x_test)
    y_test = np.argmax(y_test, 1)
    y_test = torch.LongTensor(y_test)

    softmaxes = F.softmax(model(x_test), dim=1)
    conf, predictions = torch.max(softmaxes, 1)
    determines = predictions.eq(y_test).sum().item()
    print("acc is {}".format(determines / x_test.shape[0]))

#     ind = np.random.permutation(x_test.shape[0])
#     torch.save(tuple([x_test[ind[0:1000]], y_test[ind[0:1000]]]),
#                '../data/operational.pt')
#     torch.save(tuple([x_test[ind[1000:2000]], y_test[ind[1000:2000]]]),
#                '../data/test.pt')


if __name__ == '__main__':
    model_dir = 'model.pth'
    model = Net()
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    test_model(model)
