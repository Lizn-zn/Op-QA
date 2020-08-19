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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def baseline_profit_curve(model, x_test, y_test, interval=0.05):
    softmaxes = F.softmax(model.classifier(x_test), dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    predictions = predictions.cpu().detach().numpy()
    confidences = confidences.cpu().detach().numpy()
    
    ind = np.where(confidences > 0.9)
    print('high conf mis ', np.sum((predictions != y_test.cpu().detach().numpy())[ind]))
    print('high conf correct ', np.sum((predictions == y_test.cpu().detach().numpy())[ind]))
    print(x_test.shape[0])
    print(predictions)
    print(y_test)
    print(np.sum(predictions == y_test.cpu().detach().numpy()))

    x_axis = []
    y_axis = []
    for i in range(0, 21):
        lamda = i * interval
        index = np.where(confidences >= lamda)
        indicate1 = (predictions == y_test.cpu().detach().numpy())
        indicate2 = (predictions != y_test.cpu().detach().numpy())
        ell1 = (1 - lamda) * indicate1[index] - (lamda) * indicate2[index]
        ell2 = indicate1 * (1 - lamda)
        ell1 = np.sum(ell1) / x_test.shape[0]
        print(lamda, ell1)
        ell2 = np.sum(ell2) / x_test.shape[0]
        y_axis.append(ell2 - ell1)
        x_axis.append(lamda)
    plt.plot(x_axis, y_axis, 'r')
    print(y_axis)
    np.savetxt('exp_results/baseline.csv', y_axis)


if __name__ == '__main__':
    # load original model
    BackBone = tv.models.inception_v3(
        pretrained=True, transform_input=True)
    # BackBone = torchvision.models.resnet152(pretrained=True)
    net = Net()
    net.cuda()

    # load operational test data
    x_test, y_test = torch.load('./data/training.pt')
    x_test, y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()

    baseline_profit_curve(net, x_test, y_test, interval=0.05)
