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

device = torch.device("cpu")


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


def train_model(size):
    EPOCH = 30
    BATCH_SIZE = 32

    x_test, y_test = torch.load('../data/test.pt')
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    x_test, y_test = x_test.to(device), y_test.to(device)

    test_set = TensorDataset(x_test, y_test)

    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # load operational valid data
    x_valid, y_valid = torch.load('../data/operational.pt')
    x_valid = torch.FloatTensor(x_valid)[0:size]
    y_valid = torch.LongTensor(y_valid)[0:size]
    x_op, y_op = x_valid.to(device), y_valid.to(device)

    train_set = TensorDataset(x_op, y_op)

    # train batch
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model_dir = './model.pth'
    net = Net()
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net = net.cpu()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3,
                          momentum=0.9, weight_decay=5e-6)
    # optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # with torch.no_grad():
    #     mse = 0
    #     correct = 0
    #     for data in test_loader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)
    #         outputs = F.softmax(outputs, dim=1)
    #         confidences, predictions = torch.max(outputs.data, 1)
    #         deter = predictions.detach().numpy() == labels.detach().numpy()
    #         correct += np.sum(deter)
    #         mse += np.sum(np.square(deter - confidences.detach().numpy()))
    #     print('acc is ', correct / x_test.shape[0])
    #     print('mse is ', mse / x_test.shape[0])

    print('training size ', size)
    for epoch in range(EPOCH):
        net.train()
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
                # reg_loss += 1 * torch.norm(param)

            # loss = ce_loss + reg_loss
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
    return mse, np.round(correct1 / x_test.shape[0], 3), np.round(correct2 / x_op.shape[0], 3)


if __name__ == "__main__":
    ans = []
    testacc = []
    trainacc = []
    # the last model is seperately trained by: for i in range(19, 20):
    for i in range(20):
        mse, acc1, acc2 = train_model(size=50 + 50 * i)
        ans.append(mse / 1000)
        testacc.append(acc1)
        trainacc.append(acc2)
    print(ans, testacc, trainacc)
