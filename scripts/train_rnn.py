import os, random, argparse, warnings, sys
sys.path.extend(['/Users/ahmad/software/snaplab_github/neuro_rnn', '/Users/ahmad/software/snaplab_github/neuro_rnn/packages/neurogym'])
import numpy as np
from scipy.spatial import distance
import pandas as pd
from functools import partial

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from src.neural_network import train_helper
from src.utils import normalize_x, get_weight_masks_schaefer, get_file_str, get_device, get_n_threads, get_brainmap_distance, get_seq_len

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def train(config):
    # get config params
    datadir = config['datadir']
    outdir = config['outdir']
    device = config['device']
    n_threads = config['n_threads']

    # RNN model and training parameters
    hidden_size = config['hidden_size']
    n_runs = config['n_runs']
    mask_weights = config['mask_weights']

    # regularization parameters
    kernel_type = config['kernel_type']

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
        config['masks'] = masks
        config['centroids'] = centroids
    else:
        n_io = 'na'
    config['n_io'] = n_io

    # setup regularization kernel
    if kernel_type is None:
        # no distance penalty
        regularization_kernel = None
        distance_matrix = None
    elif kernel_type == 'euclidean':
        # static distance matrix for regularization
        centroids = pd.read_csv(os.path.join(datadir, 'schaefer{0}_centroids.csv'.format(hidden_size * 2)))
        centroids = centroids[:hidden_size]  # pull out left hemisphere
        centroids.set_index("ROI Name", inplace=True)
        distance_matrix = distance.pdist(centroids, "euclidean")  # get euclidean distances between nodes
        distance_matrix = distance.squareform(distance_matrix)  # reshape to square matrix
        regularization_kernel = normalize_x(distance_matrix)
    elif kernel_type == 'sa_axis':
        brain_map = np.load(os.path.join(datadir, 'schaefer{0}_sa-axis.npy'.format(hidden_size * 2)))
        brain_map = brain_map[:hidden_size]  # pull out left hemisphere
        distance_matrix = get_brainmap_distance(brain_map=brain_map)
        regularization_kernel = normalize_x(distance_matrix)
    elif kernel_type == 'sf_axis':
        brain_map = np.load(os.path.join(datadir, 'schaefer{0}_cyto.npy'.format(hidden_size * 2)))
        brain_map = brain_map[:hidden_size]  # pull out left hemisphere
        distance_matrix = get_brainmap_distance(brain_map=brain_map)
        regularization_kernel = normalize_x(distance_matrix)
    elif kernel_type == 'struct_conn':
        conn_reg_mat = np.load(os.path.join(datadir, 'schaefer{0}_structural_conn_kernel.npy'.format(hidden_size * 2)))
        distance_matrix = conn_reg_mat[:hidden_size, :][:, :hidden_size]  # pull out left hemisphere
        regularization_kernel = normalize_x(distance_matrix)
    config['distance_matrix'] = distance_matrix
    config['regularization_kernel'] = regularization_kernel

    # get file name
    file_str = get_file_str(config)
    print('\n')
    print(file_str)
    
    # skip if outputs exist
    if os.path.isfile(os.path.join(config['outdir'], file_str + '.pt')):
        print('found outputs! skipping... ')
    else:
        # prepare partial function for multiprocessing
        partial_train_helper = partial(train_helper, config=config)
        if str(device) == 'cuda' or n_threads == 1:
            print('running in serial...')
            # initialise outputs list
            training_outputs = []
            trained_models = []
            # train runs in sequence
            for run in np.arange(n_runs):
                outputs, models = partial_train_helper(run)
                training_outputs.append(outputs)
                trained_models.append(models)
        else:
            print('running in parallel...')
            # train runs in parallel on cpu
            with torch.multiprocessing.Pool(processes=n_threads, maxtasksperchild=1) as pool:
                training_outputs, trained_models = zip(*pool.map(partial_train_helper, np.arange(n_runs)))

        # save models
        torch.save(trained_models, os.path.join(outdir, file_str + '.pt'))
        
        # save outputs
        np.save(os.path.join(outdir, file_str), training_outputs)
        
        # save config
        np.save(os.path.join(outdir, file_str + '_config'), config)
        
        # log_args = {
        #     'training_loss': training_loss,
        #     'validation_loss': validation_loss,
        #     'test_accuracy': test_accuracy,
        #     'inputs': inputs,
        #     'labels': labels,
        #     'hidden_activity': hidden_activity,
        #     'output_activity': output_activity,
        #     'info': info
        # }
        # np.save(os.path.join(outdir, file_str), log_args)


def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    # file locations
    parser.add_argument('--datadir', type=str, default='/home/lindenmp/research_projects/neuro_rnn/data')
    parser.add_argument('--outdir', type=str, default='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/pytorch/model')
    
    # device settings
    parser.add_argument('--device', type=str, default='None')
    parser.add_argument('--n_threads', type=int, default=None)

    # data parameters
    parser.add_argument('--task', type=str, default='PerceptualDecisionMaking-v0')
    parser.add_argument('--dt', type=int, default=100)
    parser.add_argument('--seq_len_multi', type=int, default=5)
    parser.add_argument('--decision', type=int, default=300)
    parser.add_argument('--standardize_task', type=str, default='False')

    # RNN model and training parameters
    parser.add_argument('--rnn_model', type=str, default='rnn-tanh')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_runs', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--epoch_log', type=int, default=100)
    parser.add_argument('--mask_weights', type=str, default='True')
 
    # regularization parameters
    parser.add_argument('--reg_type', type=str, default='l2')
    parser.add_argument('--reg_weight', type=float, default=0.001)
    parser.add_argument('--kernel_type', type=str, default='None')

    args = parser.parse_args()
    args.datadir = os.path.expanduser(args.datadir)
    args.outdir = os.path.expanduser(args.outdir)

    return args


if __name__ == '__main__':
    
    args = get_args()
    
    # Device configuration
    device = get_device(args.device)
    if str(args.device) == 'cpu':
        args.n_threads = get_n_threads(args.n_threads,1)

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
        
        # file locations
        'datadir': args.datadir,
        'outdir': args.outdir,

        # task parameters
        'task': args.task,
        'dt': args.dt,
        'seq_len_multi': args.seq_len_multi,
        'seq_len': get_seq_len(args.task, args.decision, args.seq_len_multi),

        # RNN model and training parameters
        'rnn_model': args.rnn_model,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_runs': args.n_runs,
        'n_epochs': args.n_epochs,
        'epoch_log': args.epoch_log,
        'mask_weights': args.mask_weights,

        # regularization parameters
        'reg_type': args.reg_type,
        'reg_weight': args.reg_weight,
        'kernel_type': args.kernel_type,

        'env_kwargs': env_kwargs,
        
        # device settings
        'device': args.device,
        'n_threads': args.n_threads
        
    }

    train(config=config)
