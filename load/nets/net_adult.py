import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


import warnings
warnings.filterwarnings('ignore')

_in_ = 97
_out_ = 2

def training_param(model):
    batch_size = 32
    num_epochs = 250
    learning_rate = 0.001
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
    elif net_name=="regularizedNN":
        return regularizedNN()
    elif net_name=="residualNN":
        return residualNN()
    elif net_name=="bottleneckNN":
        return bottleneckNN()
    else:
        raise ValueError(f"There is no object of class {net_name}.")

###############################################################################################################
class smallNN(nn.Module): 
    def __init__(self):        
        num_features, size1, size2, size3, size4, size5 = _in_, 256, 128, 64, 32,  _out_
        
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.sigmoid1=nn.ReLU()
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.sigmoid2=nn.ReLU()
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.sigmoid3 =nn.ReLU()
        self.linear4=nn.Linear(in_features=size3, out_features=size4)
        self.sigmoid4 =nn.ReLU()
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
        num_features, size1, size2, size3, size4, size5, size6 = _in_, 1024, 512, 128, 64, 32, _out_
        
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.relu1=nn.ReLU()
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.relu3 =nn.ReLU()
        self.linear4=nn.Linear(in_features=size3, out_features=size4)
        self.relu4 =nn.ReLU()
        self.linear5=nn.Linear(in_features=size4, out_features=size5)
        self.relu5 =nn.ReLU()
        self.linear6=nn.Linear(in_features=size5, out_features=size6)
        self.softmax = nn.Softmax()
        
    def forward(self, X):
        out=self.linear1(X)
        out=self.relu1(out)
        out=self.linear2(out)
        out=self.relu2(out)
        out=self.linear3(out)
        out=self.relu3(out)
        out=self.linear4(out)
        out=self.relu4(out)
        out=self.linear5(out)
        out=self.relu5(out)
        out=self.linear6(out)
        return self.softmax(out)


###############################################################################################################
class shallowNN(nn.Module): 
    def __init__(self):        
        num_features, size1, size2, size3  = _in_, 64, 64, _out_
        
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.relu1=nn.ReLU()
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(in_features=size2, out_features=size3)

        self.softmax = nn.Softmax()
        
    def forward(self, X):
        out=self.linear1(X)
        out=self.relu1(out)
        out=self.linear2(out)
        out=self.relu2(out)
        out=self.linear3(out)
        return self.softmax(out)
    
#################################################


class regularizedNN(nn.Module):
    def __init__(self):
        dropout_rate = 0.3
        num_features, size1, size2, size3, size4, size5 = _in_, 256, 128, 64, 32,  _out_
        
        super().__init__()
        self.linear1=nn.Linear(in_features=num_features, out_features=size1)
        self.bn1=nn.BatchNorm1d(size1)
        self.sigmoid1=nn.ReLU()
        self.dropout1=nn.Dropout(dropout_rate)
        self.linear2=nn.Linear(in_features=size1, out_features=size2)
        self.bn2=nn.BatchNorm1d(size2)
        self.sigmoid2=nn.ReLU()
        self.dropout2=nn.Dropout(dropout_rate)
        self.linear3=nn.Linear(in_features=size2, out_features=size3)
        self.bn3=nn.BatchNorm1d(size3)
        self.sigmoid3 =nn.ReLU()
        self.dropout3=nn.Dropout(dropout_rate)
        self.linear4=nn.Linear(in_features=size3, out_features=size4)
        self.bn4=nn.BatchNorm1d(size4)
        self.sigmoid4 =nn.ReLU()
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
        num_features, size1, size2, size3 = _in_, 64, 64, _out_
        super().__init__()
        pass
    def forward(self, X):
        pass
    
    
###############################################################################################################

class bottleneckNN(nn.Module):
    def __init__(self):
        num_features, size1, size2, size3, size4 = _in_, 64, 32, 16, _out_
        super().__init__()
        self.linear1 = nn.Linear(num_features, size1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(size1, size2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(size2, size3)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(size3, size4)
        self.relu4 = nn.ReLU()
        
        self.linear7 = nn.Linear(size4, size3)
        self.relu7 = nn.ReLU()
        self.linear8 = nn.Linear(size3, size2)
        self.relu8 = nn.ReLU()
        self.linear9 = nn.Linear(size2, size1)
        self.relu9 = nn.ReLU()
        self.linear10 = nn.Linear(size1, _out_)
        self.softmax = nn.Softmax()
        
    def forward(self, X):
        out = self.relu1(self.linear1(X))
        out = self.relu2(self.linear2(out))
        out = self.relu3(self.linear3(out))
        out = self.relu4(self.linear4(out))

        out = self.relu7(self.linear7(out))
        out = self.relu8(self.linear8(out))
        out = self.relu9(self.linear9(out))
        out = self.linear10(out)
        return self.softmax(out)
        