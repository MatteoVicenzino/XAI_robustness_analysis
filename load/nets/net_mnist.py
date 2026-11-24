import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import warnings

warnings.filterwarnings('ignore')

_in_ = 1
_out_ = 10
dropout_rate = 0.2


def training_param(model):
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.001
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    return batch_size, num_epochs, learning_rate, optimizer, loss_fn

def recover_net(net_name):
    if net_name == "CNN1":
        return CNN1()
    if net_name == "CNN1do":
        return CNN1bn()
    if net_name == "CNN1bn":
        return CNN1bn()
    else:
        raise ValueError(f"There is no object of class {net_name}.")


class CNN1(nn.Module):
    def __init__(self):
        in_channels, size1, size2, size3, size4, out = _in_, 32, 32, 64, 64, _out_

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, size1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(size1, size2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 28 -> 14
        self.conv3 = nn.Conv2d(size2, size3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(size3, size4, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 14 -> 7
        self.linear1 = nn.Linear(in_features=size4 * 7 * 7, out_features=out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.conv1(X)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.softmax(out)
        return out

class CNN1do(nn.Module):
    def __init__(self):
        in_channels, size1, size2, size3, size4, out = _in_, 32, 32, 64, 64, _out_

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, size1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(size1, size2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 28 -> 14
        self.conv3 = nn.Conv2d(size2, size3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(size3, size4, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 14 -> 7
        self.linear1 = nn.Linear(in_features=size4 * 7 * 7, out_features=out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.conv1(X)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.softmax(out)
        return out

class CNN1bn(nn.Module):
    def __init__(self):
        in_channels, size1, size2, size3, size4, out = _in_, 32, 32, 64, 64, _out_

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, size1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(size1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(size1, size2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(size2)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 28 -> 14
        self.conv3 = nn.Conv2d(size2, size3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(size3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(size3, size4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(size4)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 14 -> 7
        self.linear1 = nn.Linear(in_features=size4 * 7 * 7, out_features=out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.conv1(X)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        out = self.softmax(out)
        return out


