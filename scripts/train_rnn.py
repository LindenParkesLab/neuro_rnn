import os, random, time
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import scipy as sp
from scipy.spatial import distance
from scipy import signal
import pandas as pd
import bct
import neurogym as ngym

import torch
import torch.nn as nn
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

from src.neural_network import RNN
from src.utils import normalize_x, build_reg_ken

#%%
# parameters

# task
task = 'PerceptualDecisionMaking-v0'
# task = 'MultiSensoryIntegration-v0'
kwargs = {'dt': 100}
seq_len = 200
batch_size = 128

# RNN
# architecture
hidden_size = 200
n_epochs = 1000
num_layers = 1
# optimizer
learning_rate = 0.001
eps = 1e-08
betas = (0.9, 0.999)
# regularization
# reg_type = 'l1'
reg_type = 'l2'
reg_weight = 0.001

#%%
datadir = '/home/lindenmp/research_projects/neuro_rnn/data'
# modeldir = '/home/lindenmp/research_projects/neuro_rnn/results/models'
modeldir = '/media/lindenmp/storage/research_projects/neuro_rnn/results/pytorch/model'
if not os.path.exists(modeldir):
    os.makedirs(modeldir)

if hidden_size == 400:
    centroids = pd.read_csv(os.path.join(datadir, 'hcp_schaefer400_centroids.csv'))
else:
    centroids = pd.read_csv(os.path.join(datadir, 'hcp_schaefer200_centroids.csv'))

centroids.set_index("ROI Name", inplace=True)
centroids = centroids[:hidden_size]
print(centroids.head())

distance_matrix = distance.pdist(centroids, "euclidean")  # get euclidean distances between nodes
distance_matrix = distance.squareform(distance_matrix)  # reshape to square matrix
distance_matrix = normalize_x(distance_matrix)

#%%
# Make supervised dataset
dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=batch_size, seq_len=seq_len)
input_size = dataset.env.observation_space.shape[0]
num_classes = dataset.env.action_space.n

#%%
rnn_models = ['rnn-tanh', 'rnn-relu', 'lstm', 'gru']
# kernels = [None, 'standard', 'spotlight', 'additive', 'comet']
kernels = [None, 'euclidean', 'static', 'additive', 'comet']
runs = 50

for rnn_model in rnn_models:
    for kernel_type in kernels:
        # regularization
        if kernel_type is None:
            pass  # no distance penalty
        elif kernel_type == 'euclidean':
            # static distance matrix for regularization
            distance_tensor = torch.from_numpy(distance_matrix).type(torch.float).to(device)
        elif kernel_type == 'static':
            # dynamic distance matrix for regularization
            kernel = build_reg_ken(n_epochs=n_epochs, hidden_size=hidden_size, type='additive')
            distance_kernel = 1 - kernel[:, :, -1]
            distance_tensor = torch.from_numpy(distance_kernel).type(torch.float).to(device)
        else:
            # dynamic distance matrix for regularization
            kernel = build_reg_ken(n_epochs=n_epochs, hidden_size=hidden_size, type=kernel_type)
            distance_kernel = 1 - kernel
            distance_tensor = torch.from_numpy(distance_kernel).type(torch.float).to(device)

        for run in np.arange(runs):
            print(task, rnn_model, kernel_type, run)
            file_str = '{:}_' \
                       '{:}-{:}-{:}-{:}_' \
                       '{:}-{:}-{:}_' \
                       'run-{:}'.format(task,
                                    rnn_model, hidden_size, n_epochs, learning_rate,
                                    kernel_type, reg_type, reg_weight,
                                    run)

            if os.path.isfile(os.path.join(modeldir, file_str + '.pt')):
                print('skipping..')
            else:
                t_overall = timer()

                # seed random seed for reproducibility across runs
                random.seed(run)
                np.random.seed(run)
                torch.manual_seed(run)
                torch.cuda.manual_seed(run)
                torch.cuda.manual_seed_all(run)

                # instantiate model
                model = RNN(input_size, hidden_size, num_layers, num_classes, type=rnn_model).to(device)
                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps)

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
                    # outputs
                    outputs = model(inputs)
                    # loss
                    loss = criterion(outputs.view(-1, num_classes), labels)
                    # regularization
                    reg = 0.0
                    for parameter in model.parameters():
                        if parameter.ndim == 2 and parameter.shape[0] == hidden_size and parameter.shape[1] == hidden_size:
                            if kernel_type is None:
                                reg += reg_weight * model.regularization(parameter, type=reg_type)
                            elif kernel_type == 'euclidean' or kernel_type == 'static':
                                reg += reg_weight * model.regularization(parameter, type=reg_type, matrix=distance_tensor)
                            else:
                                reg += reg_weight * model.regularization(parameter, type=reg_type, matrix=distance_tensor[:, :, epoch])
                            break
                    # add regularization component
                    loss += reg
                    # perform backward pass
                    loss.backward()
                    # perform optimization
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if epoch == 0 or epoch % 100 == 99:
                        print('{:d} loss: {:0.5f}'.format(epoch + 1, running_loss / 200))
                        running_loss = 0.0
                        # print(reg)

                t_overall = timer() - t_overall
                print('Finished Training in {0}'.format(timedelta(seconds=t_overall)))

                # save model parameters
                torch.save(model.state_dict(), os.path.join(modeldir, file_str + '.pt'))
