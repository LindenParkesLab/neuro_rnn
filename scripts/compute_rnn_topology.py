import os, random, argparse, warnings, time, multiprocessing
import numpy as np
import scipy as sp
import bct
from tqdm import tqdm
import torch
# Device configuration
device = torch.device('cpu')

from src.config import REPO_ROOT, get_paths
from src.topology import get_norm_rc, threshold_adj

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# %%
def run(config):
    indir = config['indir']
    outdir = config['outdir']
    file_prefix = config['file_prefix']

    # setup output dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    # load data
    log_args = np.load(os.path.join(indir, file_prefix + '.npy'), allow_pickle=True).item()
    n_runs, n_epochs = log_args['training_loss'].shape
    hidden_size = log_args['hidden_activity'].shape[-1]

    test_accuracy = log_args['test_accuracy']
    n_logged_epochs = test_accuracy.shape[-1]

    checkpoint = torch.load(os.path.join(indir, file_prefix + '.pt'), map_location=device)
    hidden_weights = np.zeros((n_runs, n_logged_epochs, hidden_size, hidden_size))
    for i, run in enumerate(checkpoint.keys()):
        for j, epoch in enumerate(checkpoint[run].keys()):
            hidden_weights[i, j] = checkpoint[run][epoch]['rnn.weight_hh_l0']
    del checkpoint
    
    # reduce number of epochs
    trim_epochs = config['trim_epochs']
    if trim_epochs > 0:
        test_accuracy = test_accuracy[:, 1:trim_epochs]
        hidden_weights = hidden_weights[:, 1:trim_epochs]
        n_logged_epochs = test_accuracy.shape[-1]
        n_epochs = n_logged_epochs * 100
        file_prefix = file_prefix + '_trimmed-epochs-{0}'.format(trim_epochs)
    else:
        test_accuracy = test_accuracy[:, 1:]
        hidden_weights = hidden_weights[:, 1:]
        n_logged_epochs = test_accuracy.shape[-1]
        n_epochs = n_logged_epochs * 100

    # retain internal nodes
    masks = log_args['masks']
    hidden_weights = hidden_weights[:, :, masks['bystanders'], :][:, :, :, masks['bystanders']]    
    hidden_size = hidden_weights.shape[-1]

    print(test_accuracy.shape)
    print(hidden_weights.shape)
    print(n_logged_epochs, n_epochs)
    print(file_prefix)

    if config['degree']:
        file_str = os.path.join(outdir, file_prefix + '_topology-degree.npy')
        if os.path.isfile(file_str):
            print('Found outputs! skipping... ')
        else:
            print('Computing degree...')
            start = time.time()
            degree_skewness = np.zeros((n_runs, n_logged_epochs))
            degree_trimmed_mean = np.zeros((n_runs, n_logged_epochs))
            strength_mean = np.zeros((n_runs, n_logged_epochs))

            for run in tqdm(np.arange(n_runs)):
                for epoch in np.arange(n_logged_epochs):
                    A = hidden_weights[run, epoch].copy()
                    A = threshold_adj(A, binarize=False)

                    _, _, degree = bct.degrees_dir(A)
                    x, y = np.unique(degree, return_counts=True)
                    degree_skewness[run, epoch] = sp.stats.skew(y)

                    thresh = np.quantile(degree, q=0.8)
                    degree_trimmed_mean[run, epoch] = np.nanmean(degree[degree >= thresh])

                    strength = bct.strengths_dir(A)
                    strength_mean[run, epoch] = np.nanmean(strength[degree >= thresh])

            log_args = {
                'degree_skewness': degree_skewness,
                'degree_trimmed_mean': degree_trimmed_mean,
                'strength_mean': strength_mean,
            }
            np.save(os.path.join(outdir, file_str), log_args)
            end = time.time()
            print('...done in {:.2f} seconds.'.format(end-start))

    if config['modularity']:
        file_str = os.path.join(outdir, file_prefix + '_topology-modularity.npy')
        if os.path.isfile(file_str):
            print('Found outputs! skipping... ')
        else:
            print('Computing modularity...')
            start = time.time()
            modularity_directed = np.zeros((n_runs, n_logged_epochs))
            participation_coefficient = np.zeros((n_runs, n_logged_epochs, hidden_size))

            for run in tqdm(np.arange(n_runs)):
                for epoch in np.arange(n_logged_epochs):
                    A = hidden_weights[run, epoch].copy()
                    A = threshold_adj(A, binarize=False)

                    ci, modularity_directed[run, epoch] = bct.modularity_dir(A)
                    participation_coefficient[run, epoch] = bct.participation_coef(A, ci=ci, degree='directed')

            log_args = {
                'modularity_directed': modularity_directed,
                'participation_coefficient': participation_coefficient,
            }
            np.save(os.path.join(outdir, file_str), log_args)
            end = time.time()
            print('...done in {:.2f} seconds.'.format(end-start))

    if config['richclub']:
        file_str = os.path.join(outdir, file_prefix + '_topology-richclub.npy')
        if os.path.isfile(file_str):
            print('Found outputs! skipping... ')
        else:
            print('Computing normalized weighted rich-club coefficient...')
            start = time.time()
            rich_club_outputs = dict()
            rich_club_mean = np.zeros((n_runs, n_logged_epochs))

            for run in tqdm(np.arange(n_runs)):
                # prepare function inputs
                function_inputs = np.zeros((n_logged_epochs, hidden_size, hidden_size))
                for epoch in np.arange(n_logged_epochs):
                    A = hidden_weights[run, epoch].copy()
                    A = threshold_adj(A, binarize=False)
                    function_inputs[epoch] = A

                # run epochs in parallel
                with multiprocessing.Pool() as pool:
                    results = pool.map(get_norm_rc, function_inputs)

                # collect results
                rich_club_outputs[run] = dict()
                for epoch in np.arange(n_logged_epochs):
                    # store outputs incase we need them later
                    rich_club_outputs[run][epoch] = dict()
                    rich_club_outputs[run][epoch]['R'] = results[epoch][0].copy()
                    rich_club_outputs[run][epoch]['R_perm'] = results[epoch][1].copy()
                    rich_club_outputs[run][epoch]['R_norm'] = results[epoch][2].copy()
                    rich_club_outputs[run][epoch]['p_val'] = results[epoch][3].copy()

                    rich_club_mean[run, epoch] = np.nanmean(rich_club_outputs[run][epoch]['R_norm'])

            rich_club_mean[np.isinf(rich_club_mean)] = np.nan

            log_args = {
                'rich_club_mean': rich_club_mean,
                'rich_club_outputs': rich_club_outputs,
            }
            np.save(os.path.join(outdir, file_str), log_args)
            end = time.time()
            print('...done in {:.2f} seconds.'.format(end-start))


def get_args():
    '''function to get args from command line and return the args

    Returns:
        args: args that could be used by other function
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--indir', type=str, default=None,
                        help='directory holding the trained models '
                             '(default: model_dir from paths.yaml)')
    parser.add_argument('--outdir', type=str, default=None,
                        help='where to write topology results '
                             '(default: <repo>/results/topology)')
    parser.add_argument('--file_prefix', type=str, default='task-PerceptualDecisionMaking-v0-125-400_model-rnn-tanh-100-32-0.001-50-25000_wmask-True-14-27_reg-l2-0.002-sa_axis')

    # settings
    parser.add_argument('--trim_epochs', type=int, default=0)

    # topological features
    parser.add_argument('--degree', type=bool, default=True)
    parser.add_argument('--modularity', type=bool, default=True)
    parser.add_argument('--richclub', type=bool, default=True)

    args = parser.parse_args()

    # fall back to the configured project paths (see paths.yaml)
    if args.indir is None:
        args.indir = get_paths().model_dir
    if args.outdir is None:
        args.outdir = os.path.join(REPO_ROOT, 'results', 'topology')

    args.input = os.path.expanduser(args.file_prefix)
    args.outdir = os.path.expanduser(args.outdir)

    return args


if __name__ == '__main__':
    args = get_args()

    config = {
        'indir': args.indir,
        'outdir': args.outdir,
        'file_prefix': args.file_prefix,

        # settings
        'trim_epochs': args.trim_epochs,

        # topological features
        'degree': args.degree,
        'modularity': args.modularity,
        'richclub': args.richclub,
    }

    run(config=config)
