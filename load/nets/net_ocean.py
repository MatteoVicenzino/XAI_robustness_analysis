import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


import warnings
warnings.filterwarnings('ignore')


_in_ = 8
_out_ = 6
dropout_rate = 0.1

def training_param(model):
    batch_size = 32
    num_epochs = 150
    learning_rate = 0.0005
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
 
    return batch_size, num_epochs, learning_rate, optimizer, loss_fn


def recover_net(net_name):
    if net_name == "smallNN":
        return smallNN()
    elif net_name=="deeperNN":
        return deeperNN()
    elif net_name=="shallowNN":
        return shallowNN()
    elif net_name=="regSmallNN":
        return regSmallNN()
    elif net_name=="regDeeperNN":
        return regDeeperNN()
    elif net_name=="regShallowNN":
        return regShallowNN()
    else:
        raise ValueError(f"There is no object of class {net_name}.")

###############################################################################################################
class smallNN(nn.Module): 
    def __init__(self):        
        num_features, size1, size2, size3, size4, size5 = _in_, 24, 24, 16, 16,  _out_
        
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.sigmoid1=nn.Tanh()
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.sigmoid2=nn.Tanh()
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.sigmoid3 =nn.Tanh()
        self.linear4=nn.Linear(in_features=size3, out_features=size4)
        self.sigmoid4 =nn.Tanh()
        self.linear5=nn.Linear(in_features=size4, out_features=size5)
        self.softmax = nn.Softmax()
        
    def forward(self, X):
        out=self.linear1(X)
        out=self.sigmoid1(out)
        out=self.linear2(out)
        out=self.sigmoid2(out)
        out=self.linear3(out)
        out=self.sigmoid3(out)
        out=self.linear4(out)
        out=self.sigmoid4(out)
        out=self.linear5(out)
        return self.softmax(out)
    
 ###############################################################################################################
class deeperNN(nn.Module): 
    def __init__(self):        
        num_features, size1, size2, size3, size4, size5, size6 = _in_, 128, 128, 64, 32, 16 ,  _out_
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.sigmoid1=nn.Tanh()
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.sigmoid2=nn.Tanh()
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.sigmoid3 =nn.Tanh()
        self.linear4=nn.Linear(in_features=size3, out_features=size4)
        self.sigmoid4 =nn.Tanh()
        self.linear5=nn.Linear(in_features=size4, out_features=size5)
        self.sigmoid5 =nn.Tanh()
        self.linear6=nn.Linear(in_features=size5, out_features=size6)
        self.softmax = nn.Softmax()
        
    def forward(self, X):
        out=self.linear1(X)
        out=self.sigmoid1(out)
        out=self.linear2(out)
        out=self.sigmoid2(out)
        out=self.linear3(out)
        out=self.sigmoid3(out)
        out=self.linear4(out)
        out=self.sigmoid4(out)
        out=self.linear5(out)
        out = self.sigmoid5(out)
        out= self.linear6(out)
        return self.softmax(out)
    
###############################################################################################################
class shallowNN(nn.Module): 
    def __init__(self):        
        num_features, size1, size2, size3 = _in_, 16, 16, _out_
    
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.sigmoid1=nn.Tanh()
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.sigmoid2=nn.Tanh()
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.softmax = nn.Softmax()
        
    def forward(self, X):
        out=self.linear1(X)
        out=self.sigmoid1(out)
        out=self.linear2(out)
        out=self.sigmoid2(out)
        out=self.linear3(out)
        return self.softmax(out)
    
###############################################################################################################
###############################################################################################################
###############################################################################################################

class regSmallNN(nn.Module):
    def __init__(self):
        num_features, size1, size2, size3, size4, size5 = _in_, 24, 24, 16, 16, _out_

        super().__init__()
        self.linear1 = nn.Linear(in_features=num_features, out_features=size1)
        self.bn1 = nn.BatchNorm1d(size1)
        self.sigmoid1 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=size1, out_features=size2)
        self.bn2 = nn.BatchNorm1d(size2)
        self.sigmoid2 = nn.Tanh()
        self.linear3 = nn.Linear(in_features=size2, out_features=size3)
        self.bn3 = nn.BatchNorm1d(size3)
        self.sigmoid3 = nn.Tanh()
        self.linear4 = nn.Linear(in_features=size3, out_features=size4)
        self.bn4 = nn.BatchNorm1d(size4)
        self.sigmoid4 = nn.Tanh()
        self.linear5 = nn.Linear(in_features=size4, out_features=size5)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        out = self.linear1(X)
        out = self.bn1(out)
        out = self.sigmoid1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.sigmoid2(out)
        out = self.dropout(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = self.sigmoid3(out)
        out = self.dropout(out)
        out = self.linear4(out)
        out = self.bn4(out)
        out = self.sigmoid4(out)
        out = self.dropout(out)
        out = self.linear5(out)
        return self.softmax(out)


###############################################################################################################
class regDeeperNN(nn.Module):
    def __init__(self):
        num_features, size1, size2, size3, size4, size5, size6 = _in_, 128, 128, 64, 32, 16, _out_
        super().__init__()
        self.linear1 = nn.Linear(in_features=num_features, out_features=size1)
        self.bn1 = nn.BatchNorm1d(size1)
        self.sigmoid1 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=size1, out_features=size2)
        self.bn2 = nn.BatchNorm1d(size2)
        self.sigmoid2 = nn.Tanh()
        self.linear3 = nn.Linear(in_features=size2, out_features=size3)
        self.bn3 = nn.BatchNorm1d(size3)
        self.sigmoid3 = nn.Tanh()
        self.linear4 = nn.Linear(in_features=size3, out_features=size4)
        self.bn4 = nn.BatchNorm1d(size4)
        self.sigmoid4 = nn.Tanh()
        self.linear5 = nn.Linear(in_features=size4, out_features=size5)
        self.bn5 = nn.BatchNorm1d(size5)
        self.sigmoid5 = nn.Tanh()
        self.linear6 = nn.Linear(in_features=size5, out_features=size6)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        out = self.linear1(X)
        out = self.bn1(out)
        out = self.sigmoid1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.sigmoid2(out)
        out = self.dropout(out)
        out = self.linear3(out)
        out = self.bn3(out)
        out = self.sigmoid3(out)
        out = self.dropout(out)
        out = self.linear4(out)
        out = self.bn4(out)
        out = self.sigmoid4(out)
        out = self.dropout(out)
        out = self.linear5(out)
        out = self.bn5(out)
        out = self.sigmoid5(out)
        out = self.dropout(out)
        out = self.linear6(out)
        return self.softmax(out)


###############################################################################################################
class regShallowNN(nn.Module):
    def __init__(self):
        num_features, size1, size2, size3 = _in_, 16, 16, _out_

        super().__init__()
        self.linear1 = nn.Linear(in_features=num_features, out_features=size1)
        self.bn1 = nn.BatchNorm1d(size1)
        self.sigmoid1 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=size1, out_features=size2)
        self.bn2 = nn.BatchNorm1d(size2)
        self.sigmoid2 = nn.Tanh()
        self.linear3 = nn.Linear(in_features=size2, out_features=size3)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, X):
        out = self.linear1(X)
        out = self.bn1(out)
        out = self.sigmoid1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out = self.sigmoid2(out)
        out = self.dropout(out)
        out = self.linear3(out)
        return self.softmax(out)

# Other previously tested nets
"""
class regularizedNN(nn.Module):
    def __init__(self):
        dropout_rate = 0.1
        num_features, size1, size2, size3, size4, size5 = _in_, 24, 24, 16, 16, _out_
        
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.bn1=nn.BatchNorm1d(size1)
        self.sigmoid1=nn.Tanh()
        self.dropout1=nn.Dropout(dropout_rate)
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.bn2=nn.BatchNorm1d(size2)
        self.sigmoid2=nn.Tanh()
        self.dropout2=nn.Dropout(dropout_rate)
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.bn3=nn.BatchNorm1d(size3)
        self.sigmoid3 =nn.Tanh()
        self.dropout3=nn.Dropout(dropout_rate)
        self.linear4=nn.Linear(in_features=size3, out_features=size4)
        self.bn4=nn.BatchNorm1d(size4)
        self.sigmoid4 =nn.Tanh()
        self.dropout4=nn.Dropout(dropout_rate)
        self.linear5=nn.Linear(in_features=size4, out_features=size5)
        self.softmax = nn.Softmax()
        
    def forward(self, X):
        out=self.linear1(X)
        out=self.bn1(out)
        out=self.sigmoid1(out)
        out=self.dropout1(out)
        out=self.linear2(out)
        out=self.bn2(out)
        out=self.sigmoid2(out)
        out=self.dropout2(out)
        out=self.linear3(out)
        out=self.bn3(out)
        out=self.sigmoid3(out)
        out=self.dropout3(out)
        out=self.linear4(out)
        out=self.bn4(out)
        out=self.sigmoid4(out)
        out=self.dropout4(out)
        out=self.linear5(out)
        return self.softmax(out)


###############################################################################################################

class residualNN(nn.Module):
    def __init__(self):        
        num_features, size1, size2, size3, size4, size5 = _in_, 24, 24, 16, 16,  _out_
        
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.sigmoid1=nn.Tanh()
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.sigmoid2=nn.Tanh()
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.sigmoid3 =nn.Tanh()
        self.linear4=nn.Linear(in_features=size3, out_features=size4)
        self.sigmoid4 =nn.Tanh()
        self.linear5=nn.Linear(in_features=size4, out_features=size5)
        self.softmax = nn.Softmax()
        
        self.proj1 = nn.Linear(num_features, size1)
        self.proj2 = nn.Linear(size1, size2)
        self.proj3 = nn.Linear(size2, size3)
        self.proj4 = nn.Linear(size3, size4)
        
    def forward(self, X):
        identity = self.proj1(X)
        out = self.linear1(X)
        out = self.sigmoid1(out)
        out = out + identity
        
        identity = self.proj2(out)
        out = self.linear2(out)
        out = self.sigmoid2(out)
        out = out + identity
        
        identity = self.proj3(out)
        out = self.linear3(out)
        out = self.sigmoid3(out)
        out = out + identity
        
        identity = self.proj4(out)
        out = self.linear4(out)
        out = self.sigmoid4(out)
        out = out + identity
        
        out = self.linear5(out)
        return self.softmax(out)


###############################################################################################################

class CNN1(nn.Module):
    def __init__(self):
        num_features, size1, size2, size3, size4 = _in_, 16, 16, 16, _out_

        super().__init__()
        
        self.conv1 = nn.Conv1d(1, size1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(size1)
        
        self.conv2 = nn.Conv1d(size1, size2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(size2)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten(1)
        self.softmax = nn.Softmax()
        
        final_lenght = num_features // 4 # 2 pooling and conv kernel_size=3
        
        self.fc1 = nn.Linear(size2 * final_lenght, size3)
        self.fc2 = nn.Linear(size3, size4)
        
    def forward(self, X):
        
        X = X.unsqueeze(1)
        X = self.pool(self.tanh(self.bn1(self.conv1(X))))
        X = self.pool(self.tanh(self.bn2(self.conv2(X))))
        
        X = self.flatten(X)
        X = self.tanh(self.fc1(X))
        X = self.fc2(X)
        return self.softmax(X)
    
###############################################################################################################

class CNN2(nn.Module):
    def __init__(self):
        num_features, size1, size2, size3, size4 = _in_, 24, 24, 16, _out_

        super().__init__()
        
        self.conv1 = nn.Conv1d(1, size1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(size1)
        
        self.conv2 = nn.Conv1d(size1, size2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(size2)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten(1)
        self.softmax = nn.Softmax()
        
        final_lenght = num_features // 4 # 2 pooling and conv kernel_size=3
        
        self.fc1 = nn.Linear(size2 * final_lenght, size3)
        self.fc2 = nn.Linear(size3, size4)
        
    def forward(self, X):
        
        X = X.unsqueeze(1)
        X = self.pool(self.tanh(self.bn1(self.conv1(X))))
        X = self.pool(self.tanh(self.bn2(self.conv2(X))))
        
        X = self.flatten(X)
        X = self.tanh(self.fc1(X))
        X = self.fc2(X)
        return self.softmax(X)
    
###############################################################################################################

class CNN3(nn.Module):
    def __init__(self):
        num_features, size1, size2, size3, size4, size5 = _in_, 128, 64, 32, 16, _out_

        super().__init__()
        
        self.conv1 = nn.Conv1d(1, size1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(size1)
        
        self.conv2 = nn.Conv1d(size1, size2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(size2)
        
        self.conv3 = nn.Conv1d(size2, size3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(size3)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten(1)
        self.softmax = nn.Softmax()
        
        final_lenght = num_features // 4 # 2 pooling and conv kernel_size=3
        
        self.fc1 = nn.Linear(size3 * final_lenght, size4)
        self.fc2 = nn.Linear(size4, size5)
        
    def forward(self, X):
        
        X = X.unsqueeze(1)
        X = self.pool(self.tanh(self.bn1(self.conv1(X))))
        X = self.pool(self.tanh(self.bn2(self.conv2(X))))
        X = self.tanh(self.bn3(self.conv3(X)))
        
        X = self.flatten(X)
        X = self.tanh(self.fc1(X))
        X = self.fc2(X)
        return self.softmax(X)
"""