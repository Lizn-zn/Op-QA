import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional as F

import numpy as np

import matplotlib.pyplot as plt

np.random.seed(666)

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# valid the valid set size and the efficiency of calibration
# random select some examples to calibrate, and show the efficiency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(model):
    x_test, y_test = torch.load('../data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)

    x_test = torch.unsqueeze(x_test, dim=1).float()
    x_test, y_test = x_test.to(device), y_test.to(device)

    softmaxes = F.softmax(model(x_test), dim=1)
    conf, predictions = torch.max(softmaxes, 1)
    determines = predictions.eq(y_test).sum().item()
    print("acc is {}".format(determines / x_test.shape[0]))


if __name__ == '__main__':
    from train_model import Net
    model_dir = 'LeNet-5.pth'
    model = Net()
    model.load_state_dict(torch.load(model_dir))
    test_model(model)
