import os, random, argparse, warnings, sys
import numpy as np
from scipy.spatial import distance
import pandas as pd
from functools import partial
import pickle

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from src.neural_network import train_helper, ModelStateManager, ModelDataManager
from src import utils

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
    kernel_normalization = config['kernel_normalization']

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
        masks = utils.get_weight_masks_schaefer(roi_names=roi_names, input_system=input_system, output_system=output_system)
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
        regularization_kernel = utils.normalize_x(distance_matrix, kernel_normalization)
    elif kernel_type == 'sa_axis':
        brain_map = np.load(os.path.join(datadir, 'schaefer{0}_sa-axis.npy'.format(hidden_size * 2)))
        brain_map = brain_map[:hidden_size]  # pull out left hemisphere
        distance_matrix = utils.get_brainmap_distance(brain_map=brain_map)
        regularization_kernel = utils.normalize_x(distance_matrix, kernel_normalization)
    elif kernel_type == 'ut_axis':
        brain_map = np.load(os.path.join(datadir, 'schaefer{0}_ut-axis.npy'.format(hidden_size * 2)))
        brain_map = brain_map[:hidden_size]  # pull out left hemisphere
        distance_matrix = utils.get_brainmap_distance(brain_map=brain_map)
        regularization_kernel = utils.normalize_x(distance_matrix, kernel_normalization)
    elif kernel_type == 'sf_axis':
        brain_map = np.load(os.path.join(datadir, 'schaefer{0}_cyto.npy'.format(hidden_size * 2)))
        brain_map = brain_map[:hidden_size]  # pull out left hemisphere
        distance_matrix = utils.get_brainmap_distance(brain_map=brain_map)
        regularization_kernel = utils.normalize_x(distance_matrix, kernel_normalization)
    elif kernel_type == 'struct_conn':
        conn_reg_mat = np.load(os.path.join(datadir, 'schaefer{0}_structural_conn_kernel.npy'.format(hidden_size * 2)))
        distance_matrix = conn_reg_mat[:hidden_size, :][:, :hidden_size]  # pull out left hemisphere
        regularization_kernel = utils.normalize_x(distance_matrix, kernel_normalization)
    config['distance_matrix'] = distance_matrix
    config['regularization_kernel'] = regularization_kernel

    # get file name
    file_str = utils.get_file_str(config)
    print('\n')
    print(file_str)

    # set file paths
    outputs_path = os.path.join(outdir, file_str + '_outputs.h5')
    models_path = os.path.join(outdir, file_str + '_models.h5')
    config_path = os.path.join(outdir, file_str + '_config.npy')

    output_manager = ModelDataManager(outputs_path)
    model_manager = ModelStateManager(models_path)
    
    # check if outputs exist
    if os.path.isfile(models_path) and os.path.isfile(outputs_path):
        print('found existing output files! checking for missing runs... ')
        n_compl_models, _, _ = model_manager.get_info()
        n_compl_outputs, _ = output_manager.get_info()
        n_compl_runs = np.min((n_compl_models,n_compl_outputs))
        print(f'found outputs for {n_compl_runs} runs')
        if n_compl_runs == n_runs:
            all_done = True
            print('training already completed! skipping...')
        else:
            all_done = False
            rem_runs = np.arange(n_compl_runs,n_runs)
            print(f'will train {len(rem_runs)} more runs')
    else:
        all_done = False
        rem_runs = np.arange(n_runs)
    
    if not all_done:
        # prepare partial function for multiprocessing
        partial_train_helper = partial(train_helper, config=config)
        if device.type == 'cuda' or ( device.type == 'cpu' and n_threads == 1 ):
            print(f'running in serial on {device.type}...')
            if device.type == 'cuda':
                print(f"each run will use {config['n_gpu']} gpus...")
            # train runs in sequence
            for run in rem_runs:
                outputs, models = partial_train_helper(run)
                # save outputs and models
                print(f'saving outputs and models for run {run+1}')
                output_manager.save_model_data(outputs, run)
                model_manager.save_model_states(models, run)
        else:
            print(f'running in parallel on {device.type} using {n_threads} threads...')
            # prepare processing chunks
            proc_chunks = np.array_split(rem_runs, np.ceil(len(rem_runs)/n_threads))
            # train runs in parallel on cpu
            for chunk in proc_chunks:
                with torch.multiprocessing.get_context('spawn').Pool(processes=len(chunk), maxtasksperchild=1) as pool:
                    outputs, models = zip(*pool.map(partial_train_helper, chunk))
                # save outputs and models
                print(f'saving outputs and models for runs {chunk+1}')
                for idx, run in enumerate(chunk):
                    output_manager.save_model_data(outputs[idx], run)
                    model_manager.save_model_states(models[idx], run)
        # save config
        np.save(config_path, config)


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

    # RNN model and training parameters
    parser.add_argument('--rnn_model', type=str, default='rnn-tanh')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_runs', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--epoch_log', type=int, default=100)
    parser.add_argument('--mask_weights', type=str, default='True')
    # parser.add_argument('--init_rnn_weights', type=lambda s: None if s.lower() == 'none' else tuple(map(float, s.split(','))), nargs=1, default=None)
    parser.add_argument('--init_rnn_weights', type=utils.parse_float_tuple, nargs='+', default=None)
 
    # regularization parameters
    parser.add_argument('--reg_type', type=str, default='l2')
    parser.add_argument('--reg_weight', type=float, default=0.001)
    parser.add_argument('--kernel_type', type=str, default='None')
    parser.add_argument('--kernel_normalization', type=str, default='mean')
    
    # continuous time parameter
    parser.add_argument('--alpha', type=float, default=0.0)

    # parse inputs
    args = parser.parse_args()
    args.datadir = os.path.expanduser(args.datadir)
    args.outdir = os.path.expanduser(args.outdir)

    return args


if __name__ == '__main__':
    
    args = get_args()
    
    # device configuration
    device = utils.get_device(args.device)
    if device.type == 'cpu':
        n_threads = utils.get_n_threads(args.n_threads, 1)
        n_gpu = 0
    else:
        # n_threads = None
        n_threads = args.n_threads
        device = utils.get_device(device_opt=args.device, n_devices=n_threads)
        n_gpu = utils.get_n_gpu()
        for ii in range(n_gpu):
            print(f'gpu {ii} -- {torch.cuda.get_device_name(ii)}')

    # kernel and mask
    if args.kernel_type == 'None':
        kernel_type = None
    else:
        kernel_type = args.kernel_type

    if args.mask_weights == 'False':
        mask_weights = False
    elif args.mask_weights == 'True':
        mask_weights = True

    # task details 
    task_with_modifier = args.task
    task_no_modifier, task_modifier = utils.get_task_modifier(task_with_modifier)
    utils.check_if_supported(task=task_no_modifier, modifier=task_modifier)
    seq_len, timing = utils.get_seq_len_and_timing(task=task_no_modifier, modifier=task_modifier, seq_len_multi=args.seq_len_multi)
    env_kwargs = {'dt': args.dt, 'timing': timing}
    extra_kwargs = utils.get_extra_task_options(task_no_modifier)
    env_kwargs.update(extra_kwargs)
    print(' ')
    print('Task:             ' + task_no_modifier)
    print('Task modifier:    ' + task_modifier)
    print('Sequence length:  ' + str(seq_len))
    print('Task options:     ' + str(env_kwargs))
    print('Timing:           ' + str(timing))
    
    # package all info into config
    config = {
        
        # file locations
        'datadir': args.datadir,
        'outdir': args.outdir,

        # task parameters
        'task_no_modifier': task_no_modifier,
        'task_modifier': task_modifier,
        'task_with_modifier': task_with_modifier,
        'dt': args.dt,
        'seq_len_multi': args.seq_len_multi,
        'seq_len': seq_len,

        # RNN model and training parameters
        'rnn_model': args.rnn_model,
        'hidden_size': args.hidden_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_runs': args.n_runs,
        'n_epochs': args.n_epochs,
        'epoch_log': args.epoch_log,
        'mask_weights': mask_weights,
        'init_rnn_weights': args.init_rnn_weights,

        # regularization parameters
        'reg_type': args.reg_type,
        'reg_weight': args.reg_weight,
        'kernel_type': kernel_type,
        'kernel_normalization': args.kernel_normalization,
        
        # continuous time parameter
        'alpha': args.alpha,

        # ngym env
        'env_kwargs': env_kwargs,
        
        # device settings
        'device': device,
        'n_threads': n_threads,
        'n_gpu': n_gpu
        
    }

    train(config=config)
