import os, random, argparse, warnings
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
from src.utils import normalize_x, build_reg_ken, get_weight_masks, get_weight_masks_schaefer, get_file_str

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def train(config):
    # get config params
    datadir = config['datadir']
    outdir = config['outdir']

    # task parameters
    dataset = config['dataset']

    # RNN model and training parameters
    rnn_model = config['rnn_model']
    hidden_size = config['hidden_size']
    lr = config['lr']
    n_runs = config['n_runs']
    n_epochs = config['n_epochs']
    mask_weights = config['mask_weights']

    # regularization parameters
    kernel_type = config['kernel_type']
    kernel_std_frac = config['kernel_std_frac']
    comet_buffer_frac = config['comet_buffer_frac']
    comet_tail_frac = config['comet_tail_frac']

    # setup output dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # weight masks
    if mask_weights:
        centroids = pd.read_csv(os.path.join(datadir, 'schaefer{0}_centroids.csv'.format(hidden_size * 2)))
        centroids = centroids[:hidden_size]  # pull out left hemisphere
        roi_names = list(centroids['ROI Name'])
        input_system = 'Vis'
        output_system = 'Default'
        masks = get_weight_masks_schaefer(roi_names=roi_names, input_system=input_system, output_system=output_system)
        n_io = '{0}-{1}'.format(np.sum(masks['input_weight_mask']), np.sum(masks['output_weight_mask']))
    else:
        n_io = 'na'
    config['n_io'] = n_io

    # setup regularization kernel
    if kernel_type is None:
        # no distance penalty
        regularization_kernel = None
    elif kernel_type == 'euclidean':
        # static distance matrix for regularization
        centroids = pd.read_csv(os.path.join(datadir, 'schaefer{0}_centroids.csv'.format(hidden_size * 2)))
        centroids = centroids[:hidden_size]  # pull out left hemisphere
        centroids.set_index("ROI Name", inplace=True)
        distance_matrix = distance.pdist(centroids, "euclidean")  # get euclidean distances between nodes
        distance_matrix = distance.squareform(distance_matrix)  # reshape to square matrix
        regularization_kernel = normalize_x(distance_matrix)
    elif kernel_type == 'sa_axis':
        sa_axis = np.load(os.path.join(datadir, 'schaefer{0}_sa-axis.npy'.format(hidden_size * 2)))
        sa_axis = sa_axis[:hidden_size]  # pull out left hemisphere
        n = len(sa_axis)
        distance_matrix = np.zeros((n, n))
        for i in np.arange(n):
            for j in np.arange(n):
                distance_matrix[i, j] = sa_axis[i] - sa_axis[j]
        distance_matrix = np.abs(distance_matrix)
        regularization_kernel = normalize_x(distance_matrix)
    elif kernel_type == 'static':
        # dynamic distance matrix for regularization
        kernel = build_reg_ken(n_epochs=hidden_size, hidden_size=hidden_size, kernel_std_frac=kernel_std_frac,
                               type='additive', comet_buffer_frac=comet_buffer_frac, comet_tail_frac=comet_tail_frac)
        regularization_kernel = 1 - kernel[:, :, -1].copy()
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
    trained_models = dict()

    dataset.env.reset(seed=0)
    dataset.env.new_trial()
    input_size = dataset.env.observation_space.shape[0]
    n_timepoints = dataset.env.gt.shape[0]
    n_classes = dataset.env.action_space.n
    n_trials = 1000

    inputs = np.zeros((n_runs, n_trials, n_timepoints, input_size))
    labels = np.zeros((n_runs, n_trials, n_timepoints))
    hidden_activity = np.zeros((n_runs, n_trials, n_timepoints, hidden_size))
    output_activity = np.zeros((n_runs, n_trials, n_timepoints, n_classes))
    info = dict()

    # get file name
    file_str = get_file_str(config)
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
        model = RNN(input_size=input_size, hidden_size=hidden_size, num_classes=n_classes,
                    type=rnn_model, regularization_kernel=regularization_kernel,
                    input_weight_mask=input_weight_mask, output_weight_mask=output_weight_mask).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = None

        # train the model
        training_loss[run, :], validation_loss[run, :], test_accuracy[run, :], \
        trained_models[run] = run_training(dataset=dataset, model=model, optimizer=optimizer,
                                           criterion=criterion, config=config, scheduler=scheduler,
                                           return_models=True)

        # get all outputs for final model
        _, inputs[run], labels[run], hidden_activity[run], output_activity[run], info[run] \
            = run_testing(dataset=dataset, model=model, n_trials=n_trials)

    # save model and outputs
    torch.save(trained_models, os.path.join(outdir, file_str + '.pt'))
    log_args = {
        'training_loss': training_loss,
        'validation_loss': validation_loss,
        'test_accuracy': test_accuracy,
        'inputs': inputs,
        'labels': labels,
        'hidden_activity': hidden_activity,
        'output_activity': output_activity,
        'info': info
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
    parser.add_argument('--decision', type=int, default=300)
    parser.add_argument('--standardize_task', type=str, default='False')

    # RNN model and training parameters
    parser.add_argument('--rnn_model', type=str, default='rnn-tanh')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_runs', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--mask_weights', type=str, default='True')

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

    if args.standardize_task == 'True':
        args.standardize_task = True
    elif args.standardize_task == 'False':
        args.standardize_task = False

    if args.standardize_task:
        timing = {'fixation': 200, 'stimulus': 1000, 'delay': 0, 'decision': args.decision}
    else:
        timing = {'decision': args.decision}
    print(timing)
    env_kwargs = {'dt': args.dt, 'timing': timing}

    config = {
        'datadir': args.datadir,
        'outdir': args.outdir,

        # task parameters
        'task': args.task,
        'dt': args.dt,
        'seq_len': args.seq_len,

        # RNN model and training parameters
        'rnn_model': args.rnn_model,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'n_runs': args.n_runs,
        'n_epochs': args.n_epochs,
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
