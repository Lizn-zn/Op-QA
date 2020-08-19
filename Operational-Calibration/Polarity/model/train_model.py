import pandas as pd
import numpy as np
import torch


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from tensorflow.contrib import learn
import tensorflow as tf
from torch.autograd import Variable

seed = 6
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

np.random.seed(seed)



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


EPOCH = 40

x_train, y_train = torch.load('../data/training.pt')
vocab_size = np.max(x_train)
y_train = np.argmax(y_train, 1)
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
train_set = TensorDataset(x_train, y_train)
# train batch
train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
    )


x_test, y_test = torch.load('../data/tar.pt')
y_test = np.argmax(y_test,1)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

test_set = TensorDataset(x_test, y_test)

# test batch
test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=64,
        shuffle=False,
    )

# with tf.Graph().as_default():
#     sess = tf.Session()
#     with sess.as_default():
#         vocab_processor = torch.load('../data/vocab_processor.pt')
#         vocab_size = len(vocab_processor.vocabulary_)
#         print(vocab_processor)


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
net.cuda()
net.train()



for epoch in range(EPOCH):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # grad = 0
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        ce_loss = criterion(outputs, labels.long())

        # reg_loss = torch.tensor(0.)
        # for param in net.parameters():
        # reg_loss += torch.norm(param)

        loss = ce_loss

        loss.backward()
        optimizer.step()

        # print loss per batch_size
        print('The epoch %d, iteration %d, loss: %.03f'
              % (epoch + 1, i + 1, loss.item()))
    # print acc per epoch
    with torch.no_grad():
        correct = 0
        total = 0

        for data in train_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum()
        print('The %d epoch, acc is：%d%%' %
              (epoch + 1, (100 * correct / total)))
        correct = 0
        total = 0
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum()
        print('The %d epoch, acc is：%d%%' %
              (epoch + 1, (100 * correct / total)))
torch.save(net.state_dict(), 'model.pth')

correct = 0
total = 0
for data in test_loader:
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.long()).sum()
print('The %d epoch, acc is：%d%%' %
      (epoch + 1, (100 * correct / total)))