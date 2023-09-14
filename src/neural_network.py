import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, type='rnn-tanh'):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if type == 'rnn-tanh':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh')
        elif type == 'rnn-relu':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='relu')
        elif type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        elif type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)


    def regularization(self, w, type='l1', matrix=None):
        if type == 'l1' and matrix is None:
            return torch.abs(w).sum()
        elif type == 'l1' and matrix is not None:
            return torch.mul(torch.abs(w), matrix).sum()
        elif type == 'l2' and matrix is None:
            return torch.square(w).sum()
        elif type == 'l2' and matrix is not None:
            return torch.mul(torch.square(w), matrix).sum()


    def forward(self, x, return_hidden=False):
        out, hidden = self.rnn(x)
        out = self.fc(out)

        if return_hidden:
            return out, hidden
        else:
            return out
