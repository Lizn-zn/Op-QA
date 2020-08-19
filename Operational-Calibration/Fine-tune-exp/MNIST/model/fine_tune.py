import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset

import numpy as np

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence
        # padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16 * 2 * 2, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(x.size()[0], -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train_model(size):
    print('training size ', size)
    EPOCH = 50
    BATCH_SIZE = 32

    x_test, y_test = torch.load('../data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    x_test = torch.unsqueeze(x_test, dim=1)
    x_test, y_test = x_test.to(device), y_test.to(device)

    test_set = TensorDataset(x_test, y_test)

    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # load operational valid data
    x_valid, y_valid = torch.load('../data/valid.pt')
    x_valid = torch.FloatTensor(x_valid)[0:size]
    y_valid = torch.LongTensor(y_valid)[0:size]
    x_valid = torch.unsqueeze(x_valid, dim=1)
    x_op, y_op = x_valid.to(device), y_valid.to(device)

    train_set = TensorDataset(x_op, y_op)

    # train batch
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model_dir = './LeNet-5.pth'
    net = Net()
    net.load_state_dict(torch.load(model_dir))
    net = net.cpu()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-6)

    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # grad = 0
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            ce_loss = criterion(outputs, labels)

            # reg_loss = torch.tensor(0.)
            # for param in net.parameters():
            # reg_loss += torch.norm(param)

            loss = ce_loss

            loss.backward()
            optimizer.step()

            # print loss per batch_size
            # print('The epoch %d, iteration %d, loss: %.03f'
            # % (epoch + 1, i + 1, loss.item()))
        # print acc per epoch
    with torch.no_grad():
        net.eval()
        mse = 0
        correct1 = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(outputs.data, 1)
            deter = predictions.detach().numpy() == labels.detach().numpy()
            correct1 += np.sum(deter)
            mse += np.sum(np.square(deter - confidences.detach().numpy()))
        print('test acc is ', correct1 / x_test.shape[0])
        print('mse is ', mse / x_test.shape[0])
    with torch.no_grad():
        correct2 = 0
        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            outputs = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(outputs.data, 1)
            deter = predictions.detach().numpy() == labels.detach().numpy()
            correct2 += np.sum(deter)
        print('train acc is ', correct2 / x_op.shape[0])

    torch.save(net.state_dict(), './fine_tune{}.pth'.format(size))
    return mse, correct1 / x_test.shape[0], correct2 / x_op.shape[0]


if __name__ == "__main__":
    ans = []
    testacc = []
    trainacc = []
    for i in range(15):
        mse, acc1, acc2 = train_model(size=60 + 60 * i)
        ans.append(mse / 900)
        testacc.append(acc1)
        trainacc.append(acc2)
    print(ans)
    print(testacc, trainacc)
