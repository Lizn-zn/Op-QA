import os

import csv
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from PIL import Image
from torchvision import datasets, models, transforms
import torch

class ImageCLEF(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super(ImageCLEF, self).__init__()
        self.transform = transform
        file_name = root_dir + 'list/' + domain + 'List.txt'
        lines = open(file_name, 'r').readlines()
        self.images, self.labels = [], []
        self.domain = domain
        for item in lines:
            line = item.strip().split(' ')
            self.images.append(root_dir + domain + '/' + line[0].split('/')[-1])
            self.labels.append(int(line[1].strip()))

    def __getitem__(self, index):
        image = self.images[index]
        target = self.labels[index]
        img = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(img)
        return image, target

    def __len__(self):
        return len(self.images)

def load_imageclef_train(root_path, domain, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'tar': transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}
    data = ImageCLEF(root_dir=root_path, domain=domain, transform=transform_dict[phase])
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    data_train, data_val = torch.utils.data.random_split(data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=False,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, drop_last=False,
                                             num_workers=4)
    return train_loader, val_loader

def load_imageclef_test(root_path, domain, batch_size, phase):
    transform_dict = {
        'src': transforms.Compose(
            [transforms.Resize((256,256)),
             transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ]),
        'tar': transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
             ])}
    data = ImageCLEF(root_dir=root_path, domain=domain, transform=transform_dict[phase])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    return data_loader


