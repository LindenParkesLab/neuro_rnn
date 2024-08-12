import os, random, time, sys, copy
from timeit import default_timer as timer
from datetime import timedelta

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pandas as pd

import pickle
import h5py

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
    # trim = int((decision - 100) / dt)

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
            n_runs, logged_epochs = self.get_info()
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
    >>> manager.save_model_data(run)
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
