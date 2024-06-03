import os, random, time, sys, copy
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pandas as pd

from src.utils import fix_labels

import neurogym as ngym

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
            if input_weight_mask.ndim == 1:
                input_weight_mask = np.repeat(input_weight_mask[:, np.newaxis], repeats=self.input_size, axis=1)
            input_weight_mask = torch.from_numpy(input_weight_mask).type(torch.float)
            input_weight_mask = input_weight_mask.repeat(n_repeats, 1)
            self.register_buffer('input_weight_mask', input_weight_mask)
        else:
            self.input_weight_mask = input_weight_mask

        if output_weight_mask is not None:
            if output_weight_mask.ndim == 1:
                output_weight_mask = np.repeat(output_weight_mask[np.newaxis, :], repeats=self.num_classes, axis=0)
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

        hidden, h_n = self.rnn(x)
        action_pred = self.fc(hidden)

        return action_pred, hidden


def run_training(dataset, model, optimizer, criterion, config, scheduler=None, return_models=False, epoch_log=10, run=None):
    t_overall = timer()
    model.train()
    device = config['device']
    dt = config['dt']
    try:
        decision = config['env_kwargs']['timing']['decision']
        trim = int((decision - 100) / dt)
    except:
        pass
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    reg_type = config['reg_type']
    reg_weight = config['reg_weight']
    trim = int((decision - 100) / dt)

    # output container
    training_loss = []
    running_loss = 0.0
    validation_loss = []
    running_loss_val = 0.0
    test_accuracy = []
    if return_models:
        model_state = dict()

    # Train the model
    for epoch in range(n_epochs):
        # get data
        dataset.env.reset(seed=int(n_epochs+epoch))
        inputs, labels = dataset()
        try:
            labels = fix_labels(labels, decision=int(decision / dt), trim=trim)
        except:
            pass
        # split into train and validation
        inputs_tra = inputs[:, :int(batch_size/2), :]
        inputs_val = inputs[:, int(batch_size/2):, :]
        labels_tra = labels[:, :int(batch_size/2)]
        labels_val = labels[:, int(batch_size/2):]
        # convert to tensor
        inputs_tra = torch.from_numpy(inputs_tra).type(torch.float).to(device)
        inputs_val = torch.from_numpy(inputs_val).type(torch.float).to(device)
        labels_tra = torch.from_numpy(labels_tra.flatten()).type(torch.long).to(device)
        labels_val = torch.from_numpy(labels_val.flatten()).type(torch.long).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # get model outputs for training data
        outputs, _ = model(inputs_tra)
        # compute loss for training data
        loss = criterion(outputs.view(-1, model.num_classes), labels_tra)

        # perform regularization
        if model.regularization_kernel is None:
            reg = reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type)
        else:
            if model.regularization_kernel.ndim == 2:
                reg = reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type,
                                                        matrix=model.regularization_kernel)
            elif model.regularization_kernel.ndim == 3:
                reg = reg_weight * model.regularization(model.rnn.weight_hh_l0, type=reg_type,
                                                        matrix=model.regularization_kernel[:, :, epoch])
        # add regularization component
        loss += reg

        # perform backward pass
        loss.backward()
        # perform optimization
        optimizer.step()
        # update scheduler
        if scheduler is not None:
            scheduler.step()
        # store loss
        training_loss.append(loss.item())

        # validation
        with torch.no_grad():
            # get model outputs for validation data
            outputs_val, _ = model(inputs_val)
            # compute loss for validation data
            loss_val = criterion(outputs_val.view(-1, model.num_classes), labels_val)
            # store validation loss
            validation_loss.append(loss_val.item())

        # print statistics
        running_loss += loss.item()
        running_loss_val += loss_val.item()
        n_trials = 100
        if epoch == 0:
            epoch_log_timestamp_last = timer()
            model.eval()
            accuracy, _, _, _, _, _ = run_testing(dataset=dataset, model=model, n_trials=n_trials, verbose=False)
            test_accuracy.append(accuracy)
            if return_models:
                model_state[epoch] = copy.deepcopy(model.state_dict())
            model.train()
        elif epoch % epoch_log == int(epoch_log-1):
            epoch_log_timestamp_current = timer()
            epoch_log_time_elapsed = epoch_log_timestamp_current - epoch_log_timestamp_last
            epoch_log_timestamp_last = epoch_log_timestamp_current
            model.eval()
            accuracy, _, _, _, _, _ = run_testing(dataset=dataset, model=model, n_trials=n_trials, verbose=False)
            test_accuracy.append(accuracy)
            if return_models:
                model_state[epoch] = copy.deepcopy(model.state_dict())
            model.train()

            if run == None:
                print('epoch {:d} | running training loss: {:0.5f} | running validation loss: {:0.5f} | test accuracy: {:0.2f}% | time since last update: {:0.2f}s'
                    .format(epoch + 1, running_loss / epoch_log, running_loss_val / epoch_log, accuracy * 100, epoch_log_time_elapsed), flush=True)
            else:
                print('run {:d} | epoch {:d} | running training loss: {:0.5f} | running validation loss: {:0.5f} | test accuracy: {:0.2f}% | time since last update: {:0.2f}s'
                    .format(run+1, epoch + 1, running_loss / epoch_log, running_loss_val / epoch_log, accuracy * 100, epoch_log_time_elapsed), flush=True)
            running_loss = 0.0
            running_loss_val = 0.0

    t_overall = timer() - t_overall
    print('Finished training in {0}'.format(timedelta(seconds=t_overall)), flush=True)

    training_loss = np.asarray(training_loss)
    validation_loss = np.asarray(validation_loss)
    test_accuracy = np.asarray(test_accuracy)

    if return_models:
        return training_loss, validation_loss, test_accuracy, model_state
    else:
        return training_loss, validation_loss, test_accuracy


def infer_test_timing(env):
    """Infer timing of environment for testing."""
    timing = {}
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing


def run_testing(dataset, model, n_trials=1000, verbose=True):
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    elif next(model.parameters()).is_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Environment
    env = dataset.env
    env.timing = infer_test_timing(env)
    env.reset(no_step=True)

    # compute test performance
    inputs = list()
    labels = list()
    hidden_activity = list()
    output_activity = list()
    info = pd.DataFrame()
    with torch.no_grad():
        for trial in range(n_trials):
            env.reset(seed=int(trial))
            env.new_trial()
            ob, gt = env.ob, env.gt
            # print(ob.shape, gt.shape)
            inputs.append(ob)
            labels.append(gt)
            ob = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float).to(device)
            action_pred, hidden = model(ob)

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
            hidden_activity.append(np.array(hidden)[:, 0, :])
            output_activity.append(np.array(action_pred)[:, 0, :])

            # Log trial info
            trial_info = env.trial
            trial_info.update({'correct': correct, 'choice': choice})
            info = pd.concat((info, pd.DataFrame([trial_info])), ignore_index=True)

        accuracy = np.mean(info['correct'])
        if verbose:
            print('Average accuracy across {:} trials: {:.2f}%'.format(n_trials, accuracy*100), flush=True)

        # inputs = np.array(inputs)
        # labels = np.array(labels)
        # hidden_activity = np.array(hidden_activity)
        # output_activity = np.array(output_activity)

        return accuracy, inputs, labels, hidden_activity, output_activity, info


def train_helper(run, config):
    
    # create dataset
    dataset = ngym.Dataset(config['task_no_modifier'],
                           env_kwargs=config['env_kwargs'],
                           batch_size=config['batch_size'], 
                           seq_len=config['seq_len'])
    
    dataset.env.reset(seed=0)
    dataset.env.new_trial()
    input_size = dataset.env.observation_space.shape[0]
    n_classes = dataset.env.action_space.n
    hidden_size = config['hidden_size']
    n_trials = 1000
    
    # setup weight masks
    if config['mask_weights']:
        input_weight_mask = config['masks']['input_weight_mask']
        output_weight_mask = config['masks']['output_weight_mask']
    else:
        input_weight_mask = None
        output_weight_mask = None
    
    # seed random seed for reproducibility across runs
    random.seed(int(run))
    np.random.seed(int(run))
    torch.manual_seed(int(run))
    if config['device'].type == 'cuda':
        torch.cuda.manual_seed(int(run))
        torch.cuda.manual_seed_all(int(run))
    
    # initialize the model
    model = RNN(input_size=input_size, 
                hidden_size=hidden_size, 
                num_classes=n_classes, 
                type=config['rnn_model'], 
                regularization_kernel=config['regularization_kernel'], 
                input_weight_mask=input_weight_mask, 
                output_weight_mask=output_weight_mask).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    scheduler = None
    
    # train the model
    training_loss, validation_loss, test_accuracy, trained_models \
        = run_training(dataset=dataset, 
                       model=model, 
                       optimizer=optimizer,
                       criterion=criterion, 
                       config=config, 
                       scheduler=scheduler,
                       return_models=True, 
                       epoch_log=config['epoch_log'], 
                       run=run)
        
    # get all outputs for final model
    _, inputs, labels, hidden_activity, output_activity, info \
        = run_testing(dataset=dataset, 
                      model=model, 
                      n_trials=n_trials)
    
    # package all outputs into a dict
    outputs = {
                'training_loss': training_loss,
                'validation_loss': validation_loss,
                'test_accuracy': test_accuracy,
                'inputs': inputs,
                'labels': labels,
                'hidden_activity': hidden_activity,
                'output_activity': output_activity,
                'info': info,
                'run': run,
                'device': config['device']
                }
    
    return outputs, trained_models
