import os, random, time, sys, copy
from timeit import default_timer as timer
from datetime import timedelta
from typing import List, Tuple, Optional, overload, Union

import torch
from torch import Tensor
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import pickle
import h5py

from src.utils import fix_labels, get_n_gpu

import neurogym as ngym

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
                 type='rnn-tanh', regularization_kernel=None, input_weight_mask=None, output_weight_mask=None,
                 train_ih=True, train_hh=True, train_ho=True, alpha=0.0):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.sigmoid = nn.Sigmoid()
        self.rnn_type = type
        self.alpha = alpha

        if type.replace('rnn-','') in ['tanh','relu','retanh','postanh','sigmoid']:
            self.rnn = CustomRNN(input_size, hidden_size, 1, nonlinearity=type.replace('rnn-',''), alpha=alpha)
        elif type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, 1)
        elif type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, 1)
        else:
            raise ValueError(f"RNN type '{type}' not recognized.")
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
        
        if not train_ih: # freeze the input-to-hidden connections 
            nn.init.constant_(self.rnn.weight_ih_l0, 1)
            nn.init.constant_(self.rnn.bias_ih_l0, 0)
            self.rnn.weight_ih_l0.requires_grad_(False)
            self.rnn.weight_ih_l0.requires_grad_(False)
        
        if not train_hh: # freeze the hidden-to-hidden connections 
            nn.init.constant_(self.rnn.weight_hh_l0, 1)
            nn.init.constant_(self.rnn.bias_hh_l0, 0)
            self.rnn.weight_hh_l0.requires_grad_(False)
            self.rnn.weight_hh_l0.requires_grad_(False)
        
        if not train_ho: # freeze the hidden-to-output connections 
            nn.init.constant_(self.fc.weight, 1)
            nn.init.constant_(self.fc.bias, 0)
            self.fc.requires_grad_(False)

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


def rnn_custom(input: torch.Tensor, 
                hx: torch.Tensor, 
                params: torch.Tensor,
                has_biases: bool,
                num_layers: int,
                dropout: float,
                train: bool,
                bidirectional: bool,
                batch_first: bool,
                nonlinearity: str = 'postanh',
                alpha: float = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementation of RNN cell with custom activation, following PyTorch's RNN interface.
    This function broadly mimics the behavior of torch._VF.rnn_tanh and torch._VF.rnn_relu.
    
    - 'nonliearity' can be: 'tanh', 'relu', 'retanh', 'postanh', 'sigmoid'.
    - 'alpha' is a 'memory' parameter: alpha = 0 (default) yields standard RNN behavior, 
    and larger values closer to 1 cause the RNN to relie more on its previous states with each update.
    The update equation is: r(t+1) = a*r(t) + (1-a)*f(Ar(t) + Bu(t) + d).
    """
    if has_biases:
        w_ih, w_hh, b_ih, b_hh = params
    else:
        w_ih, w_hh = params
        b_ih = b_hh = None

    if batch_first and not isinstance(input, PackedSequence):
        input = input.transpose(0, 1)  # Convert to seq_len, batch, input_size

    if isinstance(input, PackedSequence):
        input_data = input.data
    else:
        input_data = input

    seq_len = input_data.size(0)
    batch_size = input_data.size(1)
    hidden_size = hx.size(-1)

    output = []
    h_n = hx  # Use this to store the final hidden state

    for t in range(seq_len):
        x_t = input_data[t]
        
        # Calculate preactivation values
        preactivation = F.linear(x_t, w_ih, b_ih) + F.linear(h_n, w_hh, b_hh)
        
        # Apply activation and continuous time update
        if nonlinearity == 'tanh':
            h_n = alpha * h_n + (1 - alpha) * torch.tanh(preactivation)
        elif nonlinearity == 'relu':
            h_n = alpha * h_n + (1 - alpha) * torch.relu(preactivation)
        elif nonlinearity == 'postanh':
            h_n = alpha * h_n + (1 - alpha) * 0.5*(torch.tanh(2*preactivation)+1)
        elif nonlinearity == 'retanh':
            h_n = alpha * h_n + (1 - alpha) * torch.relu(torch.tanh(preactivation))
        elif nonlinearity == 'sigmoid':
            h_n = alpha * h_n + (1 - alpha) * torch.sigmoid(preactivation)
        
        # Ensure h_n maintains the correct batch size
        h_n = h_n.view(batch_size, -1)
        
        output.append(h_n)
    
    # Stack sequence outputs
    output = torch.stack(output, dim=0)  # Shape: [seq_len, batch, hidden]
    
    if batch_first and not isinstance(input, PackedSequence):
        output = output.transpose(0, 1)  # Convert to batch, seq_len, hidden
    
    return output, h_n


class CustomRNN(nn.RNNBase):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = False,
                 dropout: float = 0.,
                 bidirectional: bool = False,
                 nonlinearity: str = 'postanh',
                 alpha: float = 0) -> None:
        """
        RNN module with additional activation functions and support for continous time update.
        Args broadly match PyTorch's RNN class for drop-in replacement capability with additions.
        - 'nonliearity' can be: 'tanh', 'relu', 'retanh', 'postanh', 'sigmoid'.
        - 'alpha' is a 'memory' parameter: alpha = 0 (default) yields standard RNN behavior, 
        and larger values closer to 1 cause the RNN to relie more on its previous states with each update.
        The update equation is: r(t+1) = a*r(t) + (1-a)*f(Ar(t) + Bu(t) + d).
        """
        super().__init__(
            mode='RNN_TANH',  # RNNBase only accepts RNN_TANH or RNN_RELU, but this doesn't affect our implementation because we define forward here
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.nonlinearity = nonlinearity
        self.alpha = alpha

    def forward(self, 
                input: Union[torch.Tensor, PackedSequence], 
                hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CustomRNN.
        Args match PyTorch's RNN forward method for drop-in replacement capability.
        """
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            hx = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                           max_batch_size, self.hidden_size,
                           dtype=input.dtype, device=input.device)
        else:
            # Handle hidden state reshaping if needed
            if hx.dim() == 2:
                hx = hx.unsqueeze(0)

        self.check_forward_args(input, hx, batch_sizes)
        output, hidden = rnn_custom(
            input=input,
            hx=hx[0],  # Take only the first layer for now
            params=self.all_weights[0],  # Currently only supports single layer
            has_biases=self.bias,
            num_layers=self.num_layers,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
            nonlinearity=self.nonlinearity,
            alpha=self.alpha
        )
        
        # Ensure hidden state has the correct shape
        hidden = hidden.unsqueeze(0)  # Add layer dimension
        
        return output, hidden
    

def run_training(dataset, model, optimizer, criterion, config, scheduler=None, return_models=False, epoch_log=10, run=None):
    t_overall = timer()
    model.train()
    device = config['device']
    dt = config['dt']
    try:
        decision = config['env_kwargs']['timing']['decision']
        trim = int((decision - dt) / dt)
    except:
        pass
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    reg_type = config['reg_type']
    reg_weight = config['reg_weight']

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


def run_testing_rest(model, n_steps=1000, noise_mean=0.5, noise_sd=0.3, smooth_noise=0, fix_input_channels=(0,), fix_input_value=1.0):
    if next(model.parameters()).is_cuda:
        device = torch.device('cuda')
    elif next(model.parameters()).is_mps:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # create noise input
    n_inputs = model.input_size
    inputs = np.random.normal(noise_mean, noise_sd, (n_steps,n_inputs))
    if fix_input_channels is not None:
        for i in fix_input_channels:
            inputs[:,i] = np.ones((n_steps,)) * fix_input_value
    if smooth_noise > 0:
        inputs = uniform_filter1d(inputs, size=smooth_noise, axis=0)
    ob = torch.from_numpy(inputs[:, np.newaxis, :]).type(torch.float).to(device)
    
    # test this model on noise input (resting state)
    action_pred, hidden = model(ob)
    try:
        action_pred = action_pred.detach().cpu().numpy()
        hidden = hidden.detach().cpu().numpy()
    except:
        action_pred = action_pred.detach().numpy()
        hidden = hidden.detach().numpy()
    hidden_activity = np.array(hidden)[:, 0, :]
    output_activity = np.array(action_pred)[:, 0, :]
    
    return inputs, hidden_activity, output_activity


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
                output_weight_mask=output_weight_mask,
                alpha=config['alpha'])
    # if config['n_gpu'] > 1:
    #     model = nn.DataParallel(model)
    model.to(config['device'])
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


class ModelStateManager:
    """
    Description
    -
    A class to handle writing model states to, and reading them from, a file.
    
    It allows the user to access specific model states by specifying the run and/or epoch.
    
    First initialize a new manager for the file you'd like to interact with:
    >>> manager = ModelStateManager('path/to/file.h5')
    
    Then, use one of the methods below to write to or read from that file.
    
    Methods
    -
    >>> manager.get_info()
    >>> manager.save_model_states(data, run, epoch)
    >>> manager.load_model_states(run, epoch)
    >>> manager.load_key_across_runs(epoch, key)
    """
    
    def __init__(self, filename: str):
        self.filename = filename
    
    def get_info(self):
        """
        Returns the number of runs and the indices of epochs for which data exists.

        >>> num_runs, logged_epochs, keys_per_epoch = manager.get_info()
        """
        with h5py.File(self.filename, 'r') as models_file:
            runs = list(models_file.keys())
            num_runs = len(runs)
            logged_epochs = list(int(v.replace('epoch_','')) for v in models_file[runs[0]].keys())
            logged_epochs.sort()
            keys_per_epoch = list(models_file[runs[0]][f'epoch_{logged_epochs[0]}'].keys())
        return num_runs, logged_epochs, keys_per_epoch

    def save_model_states(self, data, run=None, epoch=None):
        """
        Save model states to an HDF5 file.
        
        * To save the state from a specific run and epoch, pass that data as a dictionary: 
        
        >>> manager.save_model_states(rnn.model_state(), run=my_run, epoch=my_epoch)
        
        * To save model state dicts for all runs and epochs after training, pass that data as a list where each item is a dictionary. 
        Each dictionary should have epochs as keys, and each key should have the corresponding model state as a value.
        
        >>> manager.save_model_states(list_of_states)
        """
        
        # Function that saves the model state of a specific run and epoch.
        def save_state_for_run_and_epoch(file_path, state, run, epoch):
            with h5py.File(file_path, 'a') as models_file:
                run_group = models_file.require_group(f'run_{run}')
                epoch_group = run_group.require_group(f'epoch_{epoch}')
                for key, value in state.items():
                    if key in epoch_group:
                        del epoch_group[key]  # Remove existing dataset to avoid conflicts
                    if torch.is_tensor(value):
                        value = value.cpu().numpy()
                    epoch_group.create_dataset(key, data=value)

        # Check that run and epoch are args are compatible.
        if (run == None and (not epoch == None)):
            raise TypeError('If argument ''run'' is None, then argument ''epoch'' must also be None.')
        
        # List input is only valid if run/epoch are not defined.
        if isinstance(data,list) and (not run == None):
            raise TypeError('To save data for all runs/epochs, data must be a list.')
        
        # Save a model state from a specific run and epoch.
        if epoch is not None:
            save_state_for_run_and_epoch(self.filename, data, run, epoch)
        # Save all model states from a specific run.
        elif run is not None:
            for epoch, epoch_data in data.items():
                save_state_for_run_and_epoch(self.filename, epoch_data, run, epoch)
        # Save all model states from all runs and epochs.
        else:
            run = 0
            for run_data in data:
                for epoch, epoch_data in run_data.items():
                    save_state_for_run_and_epoch(self.filename, epoch_data, run, epoch)
                run += 1

    def load_model_states(self, run=None, epoch=None):
        """
        Load model states from an HDF5 file.
        
        * To load the model state dict from a specific run and epoch:
        
        >>> state_dict = manager.load_model_states(run, epoch)
        
        * To load model state dicts from all logged epochs of a specific run:
        
        >>> states_dicts_from_run = manager.load_model_states(run)
        
        * To load model state dicts from all runs and epochs as a list, where each list item is a dict of states from one run: 
        
        >>> state_dicts_all = manager.load_model_states()
        """
        
        # Function that loads the model state of a specific run and epoch.
        def load_state_for_run_and_epoch(file_path, run, epoch):
            with h5py.File(file_path, 'r') as models_file:
                try:
                    epoch_group = models_file[f'run_{run}/epoch_{epoch}']
                    state_dict = {key: torch.tensor(value[()]) for key, value in epoch_group.items()}
                    return state_dict
                except KeyError as e:
                    raise ValueError(f"Specified run/epoch not found: {e}")
        
        # Check that run and epoch are args are compatible.
        if (run == None and (not epoch == None)):
            raise TypeError('If argument ''run'' is None, then argument ''epoch'' must also be None.')
        
        # Load the model state of a specific run and epoch.
        if epoch is not None:
            loaded_states = load_state_for_run_and_epoch(self.filename, run, epoch)
        else:
            n_runs, logged_epochs, _ = self.get_info()
            # Load the model states of all logged epochs from a specific run.
            if run is not None:
                loaded_states = {}
                for epoch in logged_epochs:
                    loaded_states[epoch] = load_state_for_run_and_epoch(self.filename, run, epoch)
            # Load all model states for all runs and epochs.
            else:
                loaded_states = []
                for run in range(n_runs):
                    run_states = {}
                    for epoch in logged_epochs:
                        run_states[epoch] = load_state_for_run_and_epoch(self.filename, run, epoch)
                    loaded_states.append(run_states)
        
        return loaded_states

    def load_key_across_runs(self, epoch, key):
        """
        Load a specific model state key from a specific epoch across all runs.
        Returns a list (one entry per run).
        
        E.g., to load the hidden weights from all runs at epoch 99:
        >>> hidden_weights_epoch99 = manager.load_key_across_runs(epoch=99, key='weight_hh_l0')
        """        
        
        data = []
        with h5py.File(self.filename, 'r') as models_file:
            runs = [run for run in models_file.keys() if run.startswith('run_')]
            for run in runs:
                try:
                    value = models_file[f'{run}/epoch_{epoch}/{key}'][()]
                    data.append(value)
                except KeyError:
                    continue  # Skip if the epoch or key is not found in this run
        if not data:
            raise ValueError(f"No data found for epoch {epoch} and key {key}")
        return data


class ModelDataManager:
    """
    Description
    -
    A class to handle writing model performance data to, and reading them from, a file.
    
    It allows the user to access the data from a specific model by specifying the training run.
    
    First initialize a new manager for the file you'd like to interact with:
    >>> manager = ModelDataManager('path/to/file.h5')
    
    Then, use one of the methods below to write to or read from that file.
    
    Methods
    -
    >>> manager.get_info()
    >>> manager.save_model_data(data, run)
    >>> manager.load_model_data(run)
    >>> manager.load_key_across_runs(key)
    """
    
    def __init__(self, filename: str):
        self.filename = filename
    
    def get_info(self):
        """
        Returns the number of runs and the keys of the data items stored per run.

        >>> num_runs, data_keys = manager.get_info()
        """
        with h5py.File(self.filename, 'r') as models_file:
            runs = list(models_file.keys())
            num_runs = len(runs)
            data_keys = list(models_file[runs[0]].keys())
        return num_runs, data_keys

    def save_model_data(self, data, run=None):
        """
        Save model performance data to an HDF5 file.
        
        * To save the data from a specific run, pass that data as a dictionary: 
        
        >>> manager.save_model_data(run_data_dict, run=my_run)
        
        * To save model data for all runs after training, pass that data as a list where each item is a dictionary. 
        
        >>> manager.save_model_data(list_of_dicts)
        """
        
        # Function that saves the model state of a specific run.
        def save_data_for_run(file_path, data, run):
            with h5py.File(file_path, 'a') as models_file:
                run_group = models_file.require_group(f'run_{run}')
                for key, value in data.items():
                    if key in run_group:
                        del run_group[key]  # Remove existing dataset to avoid conflicts
                    # Pickle the object and store as bytes
                    pickled_data = pickle.dumps(value)
                    run_group.create_dataset(key, data=np.void(pickled_data))
                    
                    # # Store the original type as an attribute
                    # run_group[key].attrs['original_type'] = str(type(value))
        
        # List input is only valid if run is not defined.
        if isinstance(data,list) and (not run == None):
            raise TypeError('To save data for all runs, data must be a list.')
        
        # Save model data from a specific run.
        if run is not None:
            save_data_for_run(self.filename, data, run)
        # Save model data from all run.
        else:
            run = 0
            for run, run_data in enumerate(data):
                save_data_for_run(self.filename, run_data, run)

    def load_model_data(self, run=None):
        """
        Load model performance data from an HDF5 file.
        
        * To load the model data dict from a specific run:
        
        >>> data_dict = manager.load_model_data(run)
        
        * To load model data from all runs as a list where each item is a dict of data from one run: 
        
        >>> data_dicts_all = manager.load_model_data()
        """
        
        # Function that loads the model data of a specific run.
        def load_data_for_run(file_path, run):
            with h5py.File(file_path, 'r') as models_file:
                try:
                    run_group = models_file[f'run_{run}']
                    return {key: pickle.loads(value[()].tobytes()) for key, value in run_group.items()}
                except KeyError as e:
                    raise ValueError(f"Specified run not found: {e}")
        
        n_runs, _ = self.get_info()
        # Load model data for a specific run.
        if run is not None:
            loaded_data = load_data_for_run(self.filename, run)
        # Load model data for all runs.
        else:
            loaded_data = []
            for run in range(n_runs):
                loaded_data.append(load_data_for_run(self.filename, run))
        
        return loaded_data

    def load_key_across_runs(self, key):
        """
        Load a specific model performance key across all runs.
        Returns a list (one entry per run).
        
        E.g., to load the test accuracy from all runs:
        >>> test_accuracy_all = manager.load_key_across_runs('test_accuracy')
        """        
        
        data = []
        with h5py.File(self.filename, 'r') as models_file:
            runs = [run for run in models_file.keys() if run.startswith('run_')]
            for run in runs:
                try:
                    value = pickle.loads(models_file[f'{run}/{key}'][()].tobytes())
                    data.append(value)
                except KeyError:
                    continue  # Skip if the run or key is not found 
        if not data:
            raise ValueError(f"No data found for run {run} and key {key}")
        return data


def create_rnn_and_env_for_model(model_info: pd.Series, run, epoch, data_dir: str, device: torch.device):
    
    if isinstance(model_info, pd.DataFrame):
        model_info = model_info.iloc[0]
    
    # create dataset
    dataset = ngym.Dataset(model_info.task_no_modifier, 
                            env_kwargs = model_info.env_kwargs, 
                            batch_size = model_info.batch_size, 
                            seq_len = model_info.seq_len)
    dataset.env.reset(seed=0)
    dataset.env.new_trial()
    input_size = dataset.env.observation_space.shape[0]
    n_classes = dataset.env.action_space.n
    
    # load model state
    file = os.path.join(data_dir, model_info.file_str_models)
    manager = ModelStateManager(file)
    state = manager.load_model_states(run, epoch)
    
    # create rnn
    if 'regularization_kernel' in state.keys():
        regularization_kernel = np.zeros((model_info.hidden_size,model_info.hidden_size))
    else:
        regularization_kernel = None
    if 'alpha' in model_info.keys():
        alpha = model_info.alpha
    else:
        alpha = 0
    rnn = RNN(input_size = input_size, 
              hidden_size = model_info.hidden_size.item(), 
              num_classes = n_classes, 
              type = model_info.rnn_model, 
              regularization_kernel = regularization_kernel,
              input_weight_mask = np.zeros((model_info.hidden_size,input_size)), 
              output_weight_mask = np.zeros((n_classes,model_info.hidden_size)),
              alpha=alpha).to(device)
    rnn.load_state_dict(state)
    rnn.eval()
    
    return dataset, rnn
