import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, act = 'relu'):
        super(mlp, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_layers[0]
        self.hidden_dim2 = hidden_layers[-1]
        self.output_dim = output_dim
        self.layer1 = nn.Linear(input_dim, self.hidden_dim1)
        if act == 'tanh':
            self.fc1 = F.tanh
        else:
            self.fc1 = F.relu
        self.layer2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.big = len(hidden_layers) > 2
        if self.big:
            if act == 'tanh':
                self.fc_ = F.tanh
            else:
                self.fc_ = F.relu
            self.layer_ = nn.Linear(self.hidden_dim2, self.hidden_dim2)
        if act == 'tanh':
            self.fc2 = F.tanh
        else:
            self.fc2 = F.relu
        self.layer3 = nn.Linear(self.hidden_dim2, output_dim)


    def forward(self, state):
        h_s =  self.fc1(self.layer1(state))
        if self.big:
            h_s =  self.fc_(self.layer_(h_s))
        h_s =  self.fc2(self.layer2(h_s))
        return self.layer3(h_s)