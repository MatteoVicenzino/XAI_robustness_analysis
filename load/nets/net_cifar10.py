import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import warnings

warnings.filterwarnings('ignore')

_in_ = 3
_out_ = 10
dropout_rate = 0.2


def training_param(model):
    batch_size = 128
    num_epochs = 60
    learning_rate = 0.001
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    return batch_size, num_epochs, learning_rate, optimizer, loss_fn

def recover_net(net_name):
    if net_name=="CNN2":
        return CNN2()
    else:
        raise ValueError(f"There is no object of class {net_name}.")



class CNN2(nn.Module):
    def __init__(self):
        in_channels, size1, size2, size3, size4, size5, size6, size7, out = \
            _in_, 64, 64, 128, 128, 256, 256, 512, _out_

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, size1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(size1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(size1, size2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(size2)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 32 -> 16

        self.conv3 = nn.Conv2d(size2, size3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(size3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(size3, size4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(size4)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 16 -> 8

        self.conv5 = nn.Conv2d(size4, size5, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(size5)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(size5, size6, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(size6)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2) # 8 -> 4

        self.conv7 = nn.Conv2d(size6, size7, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(size7)
        self.relu7 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2) # 4 -> 2

        self.linear1 = nn.Linear(in_features=size7 * 2 * 2, out_features=out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.relu1(self.bn1(self.conv1(X)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.pool1(out)
        out = self.relu3(self.bn3(self.conv3(out)))
        out = self.relu4(self.bn4(self.conv4(out)))
        out = self.pool2(out)
        out = self.relu5(self.bn5(self.conv5(out)))
        out = self.relu6(self.bn6(self.conv6(out)))
        out = self.pool3(out)
        out = self.relu7(self.bn7(self.conv7(out)))
        out = self.pool4(out)
        out = torch.flatten(out, 1)
        out = self.linear1(out)
        return self.softmax(out)