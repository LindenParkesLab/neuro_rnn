import os, random, argparse
import numpy as np
from scipy.spatial import distance
import pandas as pd
import neurogym as ngym
from tqdm import tqdm

import torch
import torch.nn as nn
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.neural_network import RNN, run_training, run_testing
from src.utils import normalize_x, build_reg_ken

# %%
def train(dataset, args):
    # setup output dir
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # get basic task params
    input_size = dataset.env.observation_space.shape[0]
    num_classes = dataset.env.action_space.n

    # setup regularization kernel
    if args.kernel_type is None:
        # no distance penalty
        distance_tensor = None
    elif args.kernel_type == 'euclidean':
        # static distance matrix for regularization
        if args.hidden_size == 400:
            centroids = pd.read_csv(os.path.join(args.datadir, 'hcp_schaefer400_centroids.csv'))
        else:
            centroids = pd.read_csv(os.path.join(args.datadir, 'hcp_schaefer200_centroids.csv'))

        centroids.set_index("ROI Name", inplace=True)
        centroids = centroids[:args.hidden_size]
        distance_matrix = distance.pdist(centroids, "euclidean")  # get euclidean distances between nodes
        distance_matrix = distance.squareform(distance_matrix)  # reshape to square matrix
        distance_matrix = normalize_x(distance_matrix)

        distance_tensor = torch.from_numpy(distance_matrix).type(torch.float).to(device)
    elif args.kernel_type == 'static':
        # dynamic distance matrix for regularization
        kernel = build_reg_ken(n_epochs=args.n_epochs, hidden_size=args.hidden_size, kernel_std_frac=args.kernel_std_frac,
                               type='additive', comet_buffer_frac=args.comet_buffer_frac, comet_tail_frac=args.comet_tail_frac)
        distance_kernel = 1 - kernel[:, :, -1]
        distance_tensor = torch.from_numpy(distance_kernel).type(torch.float).to(device)
    else:
        # dynamic distance matrix for regularization
        kernel = build_reg_ken(n_epochs=args.n_epochs, hidden_size=args.hidden_size, kernel_std_frac=args.kernel_std_frac,
                               type=args.kernel_type, comet_buffer_frac=args.comet_buffer_frac, comet_tail_frac=args.comet_tail_frac)
        distance_kernel = 1 - kernel
        distance_tensor = torch.from_numpy(distance_kernel).type(torch.float).to(device)

    # variable containers
    tra_loss = np.zeros((args.n_runs, args.n_epochs))
    tes_accuracy = np.zeros((args.n_runs, ))
    trained_models = dict()

    if args.kernel_type is None:
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}' \
            .format(args.task, args.dt, args.seq_len, args.batch_size,
                    args.rnn_model, args.hidden_size, args.n_runs, args.n_epochs, args.lr,
                    args.mask_weights,
                    args.reg_type, args.reg_weight, args.kernel_type)
    elif args.kernel_type == 'comet':
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}-{:}-{:}-{:}' \
            .format(args.task, args.dt, args.seq_len, args.batch_size,
                    args.rnn_model, args.hidden_size, args.n_runs, args.n_epochs, args.lr,
                    args.mask_weights,
                    args.reg_type, args.reg_weight, args.kernel_type, args.kernel_std_frac, args.comet_buffer_frac, args.comet_tail_frac)
    else:
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}-{:}' \
            .format(args.task, args.dt, args.seq_len, args.batch_size,
                    args.rnn_model, args.hidden_size, args.n_runs, args.n_epochs, args.lr,
                    args.mask_weights,
                    args.reg_type, args.reg_weight, args.kernel_type, args.kernel_std_frac)
    print('\n')
    print(file_str)

    for run in np.arange(args.n_runs):
        print('Run {:}'.format(run))
        # seed random seed for reproducibility across runs
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        torch.cuda.manual_seed(run)
        torch.cuda.manual_seed_all(run)

        # initialize the model
        model = RNN(input_size, args.hidden_size, num_classes, type=args.rnn_model, mask_weights=args.mask_weights).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = None
        # train the model
        tra_loss[run, :] = run_training(dataset=dataset, model=model, optimizer=optimizer, criterion=criterion,
                                        scheduler=scheduler, n_epochs=args.n_epochs, reg_type=args.reg_type,
                                        reg_weight=args.reg_weight, distance_tensor=distance_tensor)
        # test model performance
        tes_accuracy[run] = run_testing(dataset=dataset, model=model, n_trials=1000)
        # store trained model
        trained_models[run] = model.state_dict()

    # save model and outputs
    torch.save(trained_models, os.path.join(args.outdir, file_str + '.pt'))
    log_args = {
        'tra_loss': tra_loss,
        'tes_accuracy': tes_accuracy
    }
    np.save(os.path.join(args.outdir, file_str), log_args)

def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='/home/lindenmp/research_projects/neuro_rnn/data')
    parser.add_argument('--outdir', type=str, default='/media/lindenmp/storage/research_projects/neuro_rnn/results/pytorch/model')

    # data parameters
    parser.add_argument('--task', type=str, default='PerceptualDecisionMaking-v0')
    parser.add_argument('--dt', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)

    # RNN model and training parameters
    parser.add_argument('--rnn_model', type=str, default='rnn-tanh')
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--n_runs', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mask_weights', type=str, default='False')

    # regularization parameters
    parser.add_argument('--reg_type', type=str, default='l2')
    parser.add_argument('--reg_weight', type=float, default=0.001)
    parser.add_argument('--kernel_type', type=str, default='None')
    parser.add_argument('--kernel_std_frac', type=float, default=0.25)
    parser.add_argument('--comet_buffer_frac', type=float, default=0.1)
    parser.add_argument('--comet_tail_frac', type=float, default=0.25)

    args = parser.parse_args()
    args.datadir = os.path.expanduser(args.datadir)
    args.outdir = os.path.expanduser(args.outdir)

    return args


if __name__ == '__main__':
    args = get_args()

    if args.kernel_type == 'None':
        args.kernel_type = None
    if args.mask_weights == 'False':
        args.mask_weights = False
    elif args.mask_weights == 'True':
        args.mask_weights = True

    kwargs = {'dt': args.dt}
    dataset = ngym.Dataset(args.task, env_kwargs=kwargs, batch_size=args.batch_size, seq_len=args.seq_len)

    train(dataset=dataset, args=args)
