import os, random, time, sys
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, type='rnn-tanh', mask_weights=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.mask_weights = mask_weights

        if type == 'rnn-tanh':
            self.rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity='tanh')
        elif type == 'rnn-relu':
            self.rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity='relu')
        elif type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, 1)
        elif type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)

        if self.mask_weights:
            frac = 0.33
            input_mask = torch.zeros((hidden_size, input_size)).bool()
            input_mask[:int(hidden_size * frac), :] = True
            self.register_buffer('input_mask', input_mask)

            output_mask = torch.zeros((num_classes, hidden_size)).bool()
            output_mask[:, int(hidden_size-(hidden_size * frac)):] = True
            # output_mask[:, int(hidden_size-(hidden_size * (1 - frac))):] = True
            self.register_buffer('output_mask', output_mask)

        # gain = 1.0
        # for m in self.modules():
        #     if isinstance(m, nn.RNN):
        #         init.xavier_uniform_(m.weight_ih_l0, gain=gain)
        #         init.xavier_uniform_(m.weight_hh_l0, gain=gain)
        #     elif isinstance(m, nn.Linear):
        #         init.xavier_uniform_(m.weight, gain=gain)


    def regularization(self, w, type='l1', matrix=None):
        if type == 'l1' and matrix is None:
            return torch.abs(w).sum()
        elif type == 'l1' and matrix is not None:
            return torch.mul(torch.abs(w), matrix).sum()
        elif type == 'l2' and matrix is None:
            return torch.square(w).sum()
        elif type == 'l2' and matrix is not None:
            return torch.mul(torch.square(w), matrix).sum()


    def forward(self, x):
        if self.mask_weights:
            with torch.no_grad():
                self.rnn.weight_ih_l0.mul_(self.input_mask)
                self.rnn.bias_ih_l0.mul_(self.input_mask[:, 0])
                self.fc.weight.mul_(self.output_mask)

        out, hidden = self.rnn(x)
        x = self.fc(out)

        return x, out


def run_training(dataset, model, optimizer, criterion, scheduler=None, n_epochs=1000, reg_type='l2', reg_weight=0.001, distance_tensor=None):
    t_overall = timer()
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # output container
    training_loss = []

    # Train the model
    running_loss = 0.0
    for epoch in range(n_epochs):
        # get data
        dataset.env.reset(seed=epoch)
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # get model outputs
        outputs = model(inputs)

        # compute loss
        loss = criterion(outputs.view(-1, model.num_classes), labels)

        # perform regularization
        reg = 0.0
        if distance_tensor is None:
            reg += reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type)
        elif distance_tensor.ndim == 2:
            reg += reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type, matrix=distance_tensor)
        elif distance_tensor.ndim == 3:
            reg += reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type, matrix=distance_tensor[:, :, epoch])
        # add regularization component
        loss += reg

        # perform backward pass
        loss.backward()
        # perform optimization
        optimizer.step()
        # update scheduler
        if scheduler is not None:
            scheduler.step()

        training_loss.append(loss.item())
        # print statistics
        running_loss += loss.item()
        if epoch % 500 == 499:
            print('epoch {:d}, running loss: {:0.5f}'.format(epoch + 1, running_loss / 500))
            running_loss = 0.0

    t_overall = timer() - t_overall
    print('Finished training in {0}'.format(timedelta(seconds=t_overall)))

    return np.asarray(training_loss)


def run_testing(dataset, model, n_trials=1000):
    # t_overall = timer()
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # compute test performance
    accuracy = 0
    for trial in range(n_trials):
        dataset.env.reset(seed=trial)
        dataset.env.new_trial()
        ob, gt = dataset.env.ob, dataset.env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
        outputs = model(inputs)
        try:
            outputs = outputs.detach().cpu().numpy()
        except:
            outputs = outputs.detach().numpy()
        outputs = np.argmax(outputs, axis=-1).squeeze()
        accuracy += gt[-1] == outputs[-1]

    accuracy /= n_trials
    print('Average accuracy across {:} trials: {:.2f}%'.format(n_trials, accuracy*100))

    # t_overall = timer() - t_overall
    # print('Finished testing in {0}'.format(timedelta(seconds=t_overall)))

    return accuracy
