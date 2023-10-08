import os, random, time, sys
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pandas as pd


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
                 type='rnn-tanh', regularization_kernel=None, input_weight_mask=None, output_weight_mask=None):
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

        if type == 'lstm':
            n_repeats = 4
        elif type == 'gru':
            n_repeats = 3
        else:
            n_repeats = 1

        if regularization_kernel is not None:
            regularization_kernel = torch.from_numpy(regularization_kernel).type(torch.float)
            if regularization_kernel.ndim == 2:
                regularization_kernel = regularization_kernel.repeat(n_repeats, 1)
            elif regularization_kernel.ndim == 3:
                regularization_kernel = regularization_kernel.repeat(n_repeats, 1, 1)
            self.register_buffer('regularization_kernel', regularization_kernel)
        else:
            self.regularization_kernel = regularization_kernel

        if input_weight_mask is not None:
            input_weight_mask = torch.from_numpy(input_weight_mask).type(torch.float)
            input_weight_mask = input_weight_mask.repeat(n_repeats, 1)
            self.register_buffer('input_weight_mask', input_weight_mask)
        else:
            self.input_weight_mask = input_weight_mask

        if output_weight_mask is not None:
            output_weight_mask = torch.from_numpy(output_weight_mask).type(torch.float)
            self.register_buffer('output_weight_mask', output_weight_mask)
        else:
            self.output_weight_mask = output_weight_mask

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
        with torch.no_grad():
            if self.input_weight_mask is not None:
                self.rnn.weight_ih_l0.mul_(self.input_weight_mask)
                self.rnn.bias_ih_l0.mul_(self.input_weight_mask[:, 0])
            if self.output_weight_mask is not None:
                self.fc.weight.mul_(self.output_weight_mask)

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
        outputs, _ = model(inputs)

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


def infer_test_timing(env):
    """Infer timing of environment for testing."""
    timing = {}
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing


def run_testing(dataset, model, n_trials=1000):
    # t_overall = timer()
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Environment
    env = dataset.env
    env.timing = infer_test_timing(env)
    env.reset(no_step=True)

    with torch.no_grad():
        # compute test performance
        activity = list()
        info = pd.DataFrame()
        for trial in range(n_trials):
            env.reset(seed=trial)
            env.new_trial()
            ob, gt = env.ob, env.gt
            inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, hidden = model(inputs)

            # detach
            try:
                action_pred = action_pred.detach().cpu().numpy()
                hidden = hidden.detach().cpu().numpy()
            except:
                action_pred = action_pred.detach().numpy()
                hidden = hidden.detach().numpy()

            # Compute performance
            choice = np.argmax(action_pred[-1, 0, :])
            correct = choice == gt[-1]

            # Log stimulus period activity
            activity.append(np.array(hidden)[:, 0, :])

            # Log trial info
            trial_info = env.trial
            trial_info.update({'correct': correct, 'choice': choice})
            info = pd.concat((info, pd.DataFrame([trial_info])), ignore_index=True)

        activity = np.array(activity)
        accuracy = np.mean(info['correct'])
        print('Average accuracy across {:} trials: {:.2f}%'.format(n_trials, accuracy*100))

        # t_overall = timer() - t_overall
        # print('Finished testing in {0}'.format(timedelta(seconds=t_overall)))

        return accuracy, activity, info
