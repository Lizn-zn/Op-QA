import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
     
    def __init__(self):   
        super(Net, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(16*2*2, 120)   
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

 
     

def train_model(file_path='./model'):
    EPOCH = 5
    BATCH_SIZE = 256

    # preprocess
    transform = transforms.ToTensor()

    # trainset
    train_set = tv.datasets.MNIST(
        root='./data/',
        train=True,
        download=True,
        transform=transform)

    
    # train batch
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        )

    # test set
    test_set = tv.datasets.MNIST(
        root='./data/',
        train=False,
        download=True,
        transform=transform)

    # test batch
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        )

    # loss function and optimization
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


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
            print('The epoch %d, iteration %d, loss: %.03f'
                  % (epoch + 1, i + 1, loss.item()))
        # print acc per epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('The %d epoch, acc is：%d%%' % (epoch + 1, (100 * correct / total)))
    torch.save(net.state_dict(), '%s/original_model.pth' % (file_path))


if __name__ == "__main__":
    train_model(file_path='./')
