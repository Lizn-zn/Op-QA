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

select_size = 445

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


def bs_score(conf, y_pred, y_truth, accuracy):
    fk = np.mean(conf)
    ok = np.mean((y_pred == y_truth))
    nk = conf.shape[0]

    o = accuracy

    reliability = nk * np.square(fk - ok)
    resolution = nk * np.square(ok - o)
    return reliability, resolution


def orig_profit_curve(model, x_test, y_test, interval=0.05):
    # get confidence
    softmaxes = F.softmax(model.fc3(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    confidences = confidences.detach().numpy()
    y_test = y_test.detach().numpy()

    index = np.where(predictions != y_test)
    print("high conf misclassified are {}".format(
        np.sum(confidences[index] > 0.9)))
    index = np.where(predictions == y_test)
    print("high conf correct are {}".format(np.sum(confidences[index] > 0.9)))

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test)
        indicate2 = (predictions != y_test)
        ell = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'b')
    np.savetxt('exp_results/original.csv', y_axis)

    select = x_test.detach().numpy()
    accuracy = np.mean(predictions == y_test)

    bin_boundaries = np.linspace(0, 1, 50 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    reliability, resolution = 0, 0
    for k in range(50):
        ind = np.where(np.logical_and(
            confidences >= bin_lowers[k], confidences < bin_uppers[k]))
        if ind[0].shape[0] == 0:
            continue
        r1, r2 = bs_score(confidences[ind],
                          predictions[ind], y_test[ind], accuracy)
        reliability += r1
        resolution += r2
    print("reliability is {0}, resolution is {1}".format(
        reliability / x_test.shape[0], resolution / x_test.shape[0]))
    deter = predictions == y_test
    print('brier score is {0}'.format(
        np.mean(np.square(deter - confidences))))


def calibrated_profit_curve(model, x_test, y_test, interval=0.05):
    softmaxes = F.softmax(model(o_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.detach().numpy()
    y_test = y_test.detach().numpy()
    confidences = confidences.detach().numpy()

    index = np.where(predictions != y_test)
    print("high conf misclassified are {}".format(
        np.sum(confidences[index] > 0.9)))
    index = np.where(predictions == y_test)
    print("high conf correct are {}".format(np.sum(confidences[index] > 0.9)))

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test)
        indicate2 = (predictions != y_test)
        ell = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        y_axis.append(np.sum(ell) / x_test.shape[0])
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'r')
    np.savetxt('exp_results/calibrate.csv', y_axis)

    accuracy = np.mean(predictions == y_test)
    bin_boundaries = np.linspace(0, 1, 50 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    reliability, resolution = 0, 0
    for k in range(50):
        ind = np.where(np.logical_and(
            confidences >= bin_lowers[k], confidences < bin_uppers[k]))
        if ind[0].shape[0] == 0:
            continue
        r1, r2 = bs_score(confidences[ind],
                          predictions[ind], y_test[ind], accuracy)
        reliability += r1
        resolution += r2
    print("reliability is {0}, resolution is {1}".format(
        reliability / x_test.shape[0], resolution / x_test.shape[0]))

    deter = predictions == y_test
    print('brier score is {0}'.format(
        np.mean(np.square(deter - confidences))))


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


if __name__ == '__main__':
    # load original model
    model_dir = 'model/model.pth'
    model = Net()
    model.load_state_dict(torch.load(model_dir, map_location='cpu'))

    # load operational test data
    x_test, y_test = torch.load('./data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    x_test, y_test = x_test.to(device), y_test.to(device)
    o_test = x_test
    x_test = model.hidden(x_test)

    # load operational valid data
    x_valid, y_valid = torch.load('./data/operational.pt')
    x_valid = torch.FloatTensor(x_valid)
    y_valid = torch.LongTensor(y_valid)
    x_op, y_op = x_valid.to(device), y_valid.to(device)
    x_op = model.hidden(x_op)

    valid_set = TensorDataset(x_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=64,
        shuffle=False,
    )

    iters = 3
    init_size = 30

    fig = plt.figure()

    orig_profit_curve(model, x_test, y_test)

    optimal_profit_curve(model,  x_test, y_test, interval=0.05)

    from temperature_scaling import ModelWithTemperature
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(valid_loader)

    calibrated_profit_curve(scaled_model, x_test, y_test, interval=0.05)

    plt.savefig('curve.eps', format='eps', dpi=1000)
