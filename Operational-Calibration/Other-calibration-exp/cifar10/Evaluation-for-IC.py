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

import sys
sys.path.append('./model')
import resnet

sys.path.append('./data/')
import test_model as tm

'''
Input selection and build gaussian model for c and r = c'/c
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8, 9'
use_cuda = torch.cuda.is_available()


seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


np.random.seed(1)


device = torch.device("cuda")

select_size = 445

stat = 1
length_scale = 1


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
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()

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
    ind = np.where(np.isnan(confidences))
    confidences[ind]=1
    print('brier score is {0}'.format(
        np.mean(np.square(deter - confidences))))


def calibrated_profit_curve(model, calibrated_model, test_loader, y_test, interval=0.05):
    y_test = y_test.cpu().detach().numpy()

    confidences = np.array([])
    predictions = np.array([])
    # load operational test data
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        logits = model(inputs)
        softmaxes = F.softmax(logits, dim=1)
        conf, pred = torch.max(softmaxes, 1)
        conf = calibrated_model.predict_proba(logits.cpu().detach().numpy())
        conf = np.max(conf, axis=1)
        confidences = np.append(confidences, conf)
        predictions = np.append(predictions, pred.cpu().detach().numpy())

    
    acc = np.sum(predictions == y_test) / y_test.shape[0]
    print('accuracy is {}'.format(acc))

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
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    _, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    indicate = (predictions == y_test.cpu().detach().numpy())

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
    net = resnet.ResNet18()
    model_dir = './model/model_best.pth'
    # model_dir = './model/fine_tune3000.pth'
    state_dict = torch.load(model_dir)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    net.cuda()

    # load operational test data
    x_test, y_test = torch.load('./data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    test_set = TensorDataset(x_test, y_test)
    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = tm.test(
        test_loader, net, criterion, use_cuda=True)


    x_test = np.array([])
    y_test = np.array([])
    # load operational test data
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(
        #    inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_test = np.append(x_test, outputs.cpu().detach().numpy())
        y_test = np.append(y_test, targets.cpu().detach().numpy())
    x_test = torch.Tensor(x_test.reshape(-1, 512)).cuda()
    y_test = torch.Tensor(y_test.reshape(-1)).cuda()

    # load operational valid data
    x_valid, y_valid = torch.load('./data/operational.pt')
    x_valid = torch.FloatTensor(x_valid)
    y_valid = torch.LongTensor(y_valid)
    x_op, y_op = x_valid.to(device), y_valid.to(device)

    valid_set = TensorDataset(x_valid, y_valid)
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=64,
        shuffle=False,
    )

    x_op = np.array([])
    y_op = np.array([])
    # load operational test data
    for inputs, targets in valid_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(
        #    inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = net.hidden(inputs)
        x_op = np.append(x_op, outputs.cpu().detach().numpy())
        y_op = np.append(y_op, targets.cpu().detach().numpy())
    x_op = torch.Tensor(x_op.reshape(-1, 512)).cuda()
    y_op = torch.Tensor(y_op.reshape(-1)).cuda()

    iters = 3
    init_size = 30

    fig = plt.figure()

    orig_profit_curve(net, x_test, y_test)

    optimal_profit_curve(net, x_test, y_test, interval=0.05)

    # from sklearn.calibration import CalibratedClassifierCV
    # from sklearn.calibration import _SigmoidCalibration
    # calibrator = _SigmoidCalibration()
    # x_train, _ = torch.max(F.softmax(net.classifier(x_op)), 1)
    # x_train = x_train.cpu().detach().numpy()
    # y_train = y_op.cpu().detach().numpy()
    # calibrator.fit(x_train, y_train)
    # print('platt scaling results: ')
    # calibrated_profit_curve(net, calibrator, test_loader, y_test, interval=0.05)

    # from sklearn.isotonic import IsotonicRegression
    # calibrator = IsotonicRegression(y_min=0, y_max=1)
    # x_train, _ = torch.max(F.softmax(net.classifier(x_op)), 1)
    # x_train = x_train.cpu().detach().numpy()
    # y_train = y_op.cpu().detach().numpy()
    # calibrator.fit(x_train, y_train)
    # print('isotonic regression results: ')
    # calibrated_profit_curve(net, calibrator, test_loader, y_test, interval=0.05)

    # platt scaling ++
    from sklearn.calibration import CalibratedClassifierCV
    calibrator = CalibratedClassifierCV(base_estimator=None, cv=None, method='sigmoid')
    x_train= net.classifier(x_op)
    x_train = x_train.cpu().detach().numpy()
    y_train = y_op.cpu().detach().numpy()
    calibrator.fit(x_train, y_train)
    print('isotonic regression results: ')
    calibrated_profit_curve(net, calibrator, test_loader, y_test, interval=0.05)

    from sklearn.calibration import CalibratedClassifierCV
    calibrator = CalibratedClassifierCV(base_estimator=None, cv=None, method='isotonic')
    x_train= net.classifier(x_op)
    x_train = x_train.cpu().detach().numpy()
    y_train = y_op.cpu().detach().numpy()
    calibrator.fit(x_train, y_train)
    print('isotonic regression results: ')
    calibrated_profit_curve(net, calibrator, test_loader, y_test, interval=0.05)

    plt.savefig('curve.eps', format='eps', dpi=1000)
