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
print(device)

from src.neural_network import RNN, run_training, run_testing
from src.utils import normalize_x, build_reg_ken, get_weight_masks

# %%
def train(config):
    # get config params
    datadir = config['datadir']
    outdir = config['outdir']

    # task parameters
    task = config['task']
    dataset = config['dataset']
    input_size = dataset.env.observation_space.shape[0]
    num_classes = dataset.env.action_space.n
    seq_len = config['seq_len']
    batch_size = config['batch_size']
    decision = config['env_kwargs']['timing']['decision']

    # RNN model and training parameters
    rnn_model = config['rnn_model']
    hidden_size = config['hidden_size']
    n_runs = config['n_runs']
    n_epochs = config['n_epochs']
    lr = config['lr']
    mask_weights = config['mask_weights']

    # regularization parameters
    reg_type = config['reg_type']
    reg_weight = config['reg_weight']
    kernel_type = config['kernel_type']
    kernel_std_frac = config['kernel_std_frac']
    comet_buffer_frac = config['comet_buffer_frac']
    comet_tail_frac = config['comet_tail_frac']

    # setup output dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # weight masks
    frac = 0.33
    masks = get_weight_masks(hidden_size=hidden_size, frac=frac)

    # setup regularization kernel
    if kernel_type is None:
        # no distance penalty
        regularization_kernel = None
    elif kernel_type == 'euclidean':
        # static distance matrix for regularization
        if hidden_size == 400:
            centroids = pd.read_csv(os.path.join(datadir, 'hcp_schaefer400_centroids.csv'))
        else:
            centroids = pd.read_csv(os.path.join(datadir, 'hcp_schaefer200_centroids.csv'))

        centroids.set_index("ROI Name", inplace=True)
        centroids = centroids[:hidden_size]
        distance_matrix = distance.pdist(centroids, "euclidean")  # get euclidean distances between nodes
        distance_matrix = distance.squareform(distance_matrix)  # reshape to square matrix
        regularization_kernel = normalize_x(distance_matrix)
    elif kernel_type == 'static':
        # dynamic distance matrix for regularization
        kernel = build_reg_ken(n_epochs=n_epochs, hidden_size=hidden_size, kernel_std_frac=kernel_std_frac,
                               type='additive', comet_buffer_frac=comet_buffer_frac, comet_tail_frac=comet_tail_frac)
        regularization_kernel = 1 - kernel[:, :, -1].copy()
        del kernel
    elif kernel_type == 'constant':
        # dynamic distance matrix for regularization
        kernel = build_reg_ken(n_epochs=n_epochs, hidden_size=hidden_size, kernel_std_frac=kernel_std_frac,
                               type='additive', comet_buffer_frac=comet_buffer_frac, comet_tail_frac=comet_tail_frac)
        regularization_kernel = 1 - kernel[:, :, -1].copy()
        min_val = np.min(regularization_kernel[masks['mask_bu']])
        regularization_kernel[:] = min_val
        del kernel
    else:
        # dynamic distance matrix for regularization
        kernel = build_reg_ken(n_epochs=n_epochs, hidden_size=hidden_size, kernel_std_frac=kernel_std_frac,
                               type=kernel_type, comet_buffer_frac=comet_buffer_frac, comet_tail_frac=comet_tail_frac)
        regularization_kernel = 1 - kernel.copy()
        del kernel

    # setup weight masks
    if mask_weights:
        input_weight_mask = masks['input_weight_mask']
        output_weight_mask = masks['output_weight_mask']
    else:
        input_weight_mask = None
        output_weight_mask = None

    # variable containers
    training_loss = np.zeros((n_runs, n_epochs))
    validation_loss = np.zeros((n_runs, n_epochs))
    epoch_log = 100
    test_accuracy = np.zeros((n_runs, int((n_epochs/epoch_log)+1)))
    n_trials = 1000
    dataset.env.reset(seed=0)
    dataset.env.new_trial()
    activity = np.zeros((n_runs, n_trials, dataset.env.gt.shape[0], hidden_size))
    trained_models = dict()

    if kernel_type is None:
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}' \
            .format(task, seq_len, batch_size, decision,
                    rnn_model, hidden_size, n_runs, n_epochs, lr,
                    mask_weights,
                    reg_type, reg_weight, kernel_type)
    elif kernel_type == 'comet':
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}-{:}-{:}-{:}' \
            .format(task, seq_len, batch_size, decision,
                    rnn_model, hidden_size, n_runs, n_epochs, lr,
                    mask_weights,
                    reg_type, reg_weight, kernel_type, kernel_std_frac, comet_buffer_frac, comet_tail_frac)
    else:
        file_str = 'task-{:}-{:}-{:}-{:}_' \
                   'model-{:}-{:}-{:}-{:}-{:}_' \
                   'wmask-{:}_' \
                   'reg-{:}-{:}-{:}-{:}' \
            .format(task, seq_len, batch_size, decision,
                    rnn_model, hidden_size, n_runs, n_epochs, lr,
                    mask_weights,
                    reg_type, reg_weight, kernel_type, kernel_std_frac)
    print('\n')
    print(file_str)

    for run in np.arange(n_runs):
        print('Run {:}'.format(run))
        # seed random seed for reproducibility across runs
        random.seed(int(run))
        np.random.seed(int(run))
        torch.manual_seed(int(run))
        torch.cuda.manual_seed(int(run))
        torch.cuda.manual_seed_all(int(run))

        # initialize the model
        model = RNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes,
                    type=rnn_model, regularization_kernel=regularization_kernel,
                    input_weight_mask=input_weight_mask, output_weight_mask=output_weight_mask).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = None
        # train the model
        training_loss[run, :], validation_loss[run, :], test_accuracy[run, :], \
        trained_models[run] = run_training(dataset=dataset, model=model, optimizer=optimizer,
                                           criterion=criterion, config=config, scheduler=scheduler, return_models=True)
        # test model performance
        _, activity[run], _ = run_testing(dataset=dataset, model=model, n_trials=n_trials)

    # save model and outputs
    torch.save(trained_models, os.path.join(outdir, file_str + '.pt'))
    log_args = {
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'test_accuracy': test_accuracy,
        'activity': activity,
    }
    np.save(os.path.join(outdir, file_str), log_args)

def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--datadir', type=str, default='/home/lindenmp/research_projects/neuro_rnn/data')
    parser.add_argument('--outdir', type=str, default='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/pytorch/model')

    # data parameters
    parser.add_argument('--task', type=str, default='PerceptualDecisionMaking-v0')
    parser.add_argument('--dt', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--decision', type=int, default=300)

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

    timing = {'decision': args.decision}
    env_kwargs = {'dt': args.dt, 'timing': timing}

    config = {
        'datadir': args.datadir,
        'outdir': args.outdir,

        # task parameters
        'task': args.task,
        'dt': args.dt,
        'seq_len': args.seq_len,
        'batch_size': args.batch_size,

        # RNN model and training parameters
        'rnn_model': args.rnn_model,
        'hidden_size': args.hidden_size,
        'n_runs': args.n_runs,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'mask_weights': args.mask_weights,

        # regularization parameters
        'reg_type': args.reg_type,
        'reg_weight': args.reg_weight,
        'kernel_type': args.kernel_type,
        'kernel_std_frac': args.kernel_std_frac,
        'comet_buffer_frac': args.comet_buffer_frac,
        'comet_tail_frac': args.comet_tail_frac,

        'env_kwargs': env_kwargs
    }

    dataset = ngym.Dataset(config['task'], env_kwargs=config['env_kwargs'],
                           batch_size=config['batch_size'], seq_len=config['seq_len'])
    config['dataset'] = dataset
    train(config=config)
