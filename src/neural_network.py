from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, type='rnn-tanh'):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        if type == 'rnn-tanh':
            self.rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity='tanh')
        elif type == 'rnn-relu':
            self.rnn = nn.RNN(input_size, hidden_size, 1, nonlinearity='relu')
        elif type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, 1)
        elif type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, 1)
        self.fc = nn.Linear(hidden_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.RNN):
                # init.xavier_uniform_(m.weight_ih_l0)
                init.xavier_uniform_(m.weight_hh_l0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)


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


def train(model_environment, n_epochs=1000):
    t_overall = timer()
    # core variables
    try:
        dataset = model_environment['dataset']
        model = model_environment['model']

        optimizer = model_environment['optimizer']
        criterion = model_environment['criterion']

        reg_type = model_environment['reg_type']
        reg_weight = model_environment['reg_weight']
        kernel_type = model_environment['kernel_type']
    except KeyError:
        print('Core variables not found')

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
        if kernel_type is None:
            reg += reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type)
        elif kernel_type == 'euclidean' or kernel_type == 'static':
            reg += reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type, matrix=model_environment['distance_tensor'])
        else:
            reg += reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type, matrix=model_environment['distance_tensor'][:, :, epoch])
        # add regularization component
        loss += reg

        # perform backward pass
        loss.backward()
        # perform optimization
        optimizer.step()
        # update scheduler
        if 'scheduler' in model_environment:
            model_environment['scheduler'].step()

        training_loss.append(loss.item())
        # print statistics
        running_loss += loss.item()
        if epoch % 100 == 99:
            print('epoch {:d}, running loss: {:0.5f}'.format(epoch + 1, running_loss / 100))
            running_loss = 0.0

    t_overall = timer() - t_overall
    print('Finished training in {0}'.format(timedelta(seconds=t_overall)))

    return np.asarray(training_loss)


def test(model_environment, n_trials=1000):
    t_overall = timer()
    # core variables
    try:
        dataset = model_environment['dataset']
        model = model_environment['model']
    except KeyError:
        print('Core variables not found')

    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # compute test performance
    performance = 0
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
        performance += gt[-1] == outputs[-1]

    performance /= n_trials
    print('Average performance in {:d} trials: {:.2f}'.format(n_trials, performance))

    t_overall = timer() - t_overall
    print('Finished testing in {0}'.format(timedelta(seconds=t_overall)))

    return performance
