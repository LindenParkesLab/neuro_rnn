"""Compute weight–kernel similarity and fMRI variance explained across training epochs.

Iterates over models × runs × sampled epochs, computes metrics at each checkpoint,
and saves results as a pickle file for downstream plotting.

Usage:
    python scripts/biornn_results_dynamics_trajectory.py \
        --model_params model_params_202603bp \
        --rows 2 3 4 5 6 7 10 11 12 13 14 15 \
        --n_epochs 10 \
        --n_trials 100 \
        --n_fmri_subj 100 \
        --n_pc 5 \
        --n_jobs -1 \
        --outdir /path/to/output/
"""

import argparse
import copy
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import torch

import src.utils as utils
from src.neural_network import (
    ModelStateManager,
    RNN,
    run_testing,
    run_testing_rest,
)
import neurogym as ngym

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def subsample_epochs(logged_epochs, n_epochs):
    """Select n_epochs evenly spaced checkpoints, always including first and last."""
    if n_epochs >= len(logged_epochs):
        return logged_epochs[:]
    indices = np.round(np.linspace(0, len(logged_epochs) - 1, n_epochs)).astype(int)
    return [logged_epochs[i] for i in sorted(set(indices))]


def fit_pca(data, n_components=3):
    """Fit PCA on time series data (list of trials or single array)."""
    if isinstance(data, list):
        n_trials = len(data)
        n_timepoints, n_nodes = data[0].shape
        data_mat = np.reshape(np.array(data), (n_trials * n_timepoints, n_nodes))
    else:
        data_mat = data

    pca = PCA(n_components=n_components)
    pca.fit(data_mat)

    variance_explained = {
        'each': pca.explained_variance_ratio_,
        'cumul': np.cumsum(pca.explained_variance_ratio_),
        'total': np.sum(pca.explained_variance_ratio_),
    }
    return pca, variance_explained


def pca_projection(data, pca_fitted, centering_mean='data'):
    """Project data into a fitted PCA space and compute variance explained."""
    if isinstance(data, list):
        n_trials = len(data)
        n_timepoints, _ = data[0].shape
        data_mat = np.reshape(np.array(data), (n_trials * n_timepoints, -1))
    else:
        data_mat = np.asarray(data)

    if centering_mean == 'data':
        data_mat = data_mat - np.mean(data_mat, axis=0)
    elif centering_mean == 'pca':
        data_mat = data_mat - pca_fitted.mean_
    else:
        raise ValueError(f"centering_mean must be 'data' or 'pca', got {centering_mean!r}")

    data_pc = data_mat @ pca_fitted.components_.T

    total_variance = np.sum(np.power(data_mat, 2))
    pc_variances = np.sum(np.power(data_pc, 2), axis=0)
    variance_explained_each = pc_variances / total_variance

    return {
        'each': variance_explained_each,
        'cumul': np.cumsum(variance_explained_each),
        'total': np.sum(variance_explained_each),
    }


def compute_weight_kernel_similarity(W_hh, kernel_similarity):
    """Compute similarity metrics between learned weights and spatial kernel."""
    include_mask = ~np.eye(W_hh.shape[0], dtype=np.bool_)
    w_vec = W_hh[include_mask]
    k_vec = kernel_similarity[include_mask]

    r_spearman, _ = stats.spearmanr(w_vec, k_vec)
    eucl_dist = np.linalg.norm(w_vec - k_vec)
    cos_sim = np.dot(w_vec, k_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(k_vec))

    return {
        'spearman': r_spearman,
        'euclidean': eucl_dist,
        'cosine': cos_sim,
    }


def compute_losses(rnn, dataset, n_trials, reg_type, reg_weight):
    """Compute loss components at a checkpoint.

    Returns a dict with:
        task_loss: cross-entropy loss (mean over trials)
        l2_component: reg_weight * sum(W_hh^2) / hidden_size (only for pearson_l2s)
        pearson_component: reg_weight * (1 - corr) (only for pearson/pearson_l2s)
        total_spatial_loss: reg_weight * regularization(W_hh, ...)
        total_loss: task_loss + total_spatial_loss
    """
    import torch.nn as nn

    W_hh = rnn.rnn.weight_hh_l0
    hidden_size = W_hh.shape[0]
    kernel = rnn.regularization_kernel  # may be None

    # task loss: run a batch through the model
    criterion = nn.CrossEntropyLoss()
    task_losses = []
    env = dataset.env
    env.reset(seed=0)
    for _ in range(n_trials):
        env.new_trial()
        obs = torch.tensor(env.ob[np.newaxis, :, :], dtype=torch.float32,
                           device=W_hh.device)
        gt = torch.tensor(env.gt[np.newaxis, :], dtype=torch.long,
                          device=W_hh.device)
        with torch.no_grad():
            outputs, _ = rnn(obs)
            tl = criterion(outputs.view(-1, rnn.num_classes), gt.view(-1))
        task_losses.append(tl.item())
    task_loss = float(np.mean(task_losses))

    # spatial loss components
    result = {
        'task_loss': task_loss,
        'l2_component': None,
        'pearson_component': None,
        'total_spatial_loss': None,
        'total_loss': None,
    }

    with torch.no_grad():
        if reg_weight == 0 or (kernel is None and reg_type in ('pearson', 'pearson_l2s')):
            result['total_spatial_loss'] = 0.0
            result['total_loss'] = task_loss
            return result

        # total spatial loss using the model's own method
        if kernel is not None:
            matrix = kernel if kernel.ndim == 2 else None
            total_spatial = reg_weight * rnn.regularization(W_hh, type=reg_type, matrix=matrix)
        else:
            total_spatial = reg_weight * rnn.regularization(W_hh, type=reg_type)
        result['total_spatial_loss'] = total_spatial.item()

        # decompose for pearson_l2s
        if reg_type == 'pearson_l2s' and kernel is not None:
            # L2 component: reg_weight * sum(W^2) / hidden_size (unweighted by matrix)
            l2_val = reg_weight * torch.square(W_hh).sum() / hidden_size
            result['l2_component'] = l2_val.item()

            # Pearson component: reg_weight * (1 - corr)
            similarity = 1.0 - kernel
            off_diag = ~torch.eye(hidden_size, dtype=torch.bool, device=W_hh.device)
            w_flat = W_hh[off_diag]
            s_flat = similarity[off_diag]
            corr = torch.corrcoef(torch.stack([w_flat, s_flat]))[0, 1]
            pearson_val = reg_weight * (1.0 - corr)
            result['pearson_component'] = pearson_val.item()

        elif reg_type == 'pearson' and kernel is not None:
            result['pearson_component'] = result['total_spatial_loss']

        result['total_loss'] = task_loss + result['total_spatial_loss']

    return result


def create_dataset_and_rnn_shell(model_info, state_template, device):
    """Create the dataset and an uninitialized RNN for a given model config.

    The RNN is constructed with the right architecture but its weights are
    not meaningful — call rnn.load_state_dict(state) before using it.
    """
    if isinstance(model_info, pd.DataFrame):
        model_info = model_info.iloc[0]

    dataset = ngym.Dataset(
        model_info.task_no_modifier,
        env_kwargs=model_info.env_kwargs,
        batch_size=model_info.batch_size,
        seq_len=model_info.seq_len)
    dataset.env.reset(seed=0)
    dataset.env.new_trial()
    input_size = dataset.env.observation_space.shape[0]
    n_classes = dataset.env.action_space.n

    regularization_kernel = (
        np.zeros((model_info.hidden_size, model_info.hidden_size))
        if 'regularization_kernel' in state_template else None)
    input_weight_mask = (
        np.zeros((model_info.hidden_size, input_size))
        if 'input_weight_mask' in state_template else None)
    output_weight_mask = (
        np.zeros((n_classes, model_info.hidden_size))
        if 'input_weight_mask' in state_template else None)
    alpha = model_info.alpha if 'alpha' in model_info.keys() else 0

    rnn = RNN(
        input_size=input_size,
        hidden_size=model_info.hidden_size.item(),
        num_classes=n_classes,
        type=model_info.rnn_model,
        regularization_kernel=regularization_kernel,
        input_weight_mask=input_weight_mask,
        output_weight_mask=output_weight_mask,
        alpha=alpha,
    ).to(device)

    return dataset, rnn


def process_one_run(run_idx, epoch_states, sampled_epochs, model_info,
                    rnn_template, dataset, has_kernel, kernel_sim,
                    reg_type, reg_weight,
                    n_trials, n_pc, fmri_rest_nsteps,
                    fmri_task_ts, fmri_rest_ts, n_fmri_subj, skip_fmri, device):
    """Process all epochs for a single run. Designed to run in a worker process."""
    torch.set_num_threads(1)
    warnings.simplefilter(action='ignore', category=UserWarning)

    rnn = copy.deepcopy(rnn_template)
    run_result = {'epochs': {}}

    for epoch in sampled_epochs:
        state = epoch_states[epoch]
        rnn.load_state_dict(state)
        rnn.eval()

        epoch_result = {}

        # weight-kernel similarity
        if has_kernel:
            W_hh = rnn.rnn.weight_hh_l0.detach().cpu().numpy()
            epoch_result['weight_kernel_similarity'] = \
                compute_weight_kernel_similarity(W_hh, kernel_sim)
        else:
            epoch_result['weight_kernel_similarity'] = None

        # losses
        epoch_result['losses'] = compute_losses(
            rnn, dataset, n_trials, reg_type, reg_weight)

        # task inference
        accuracy, _, _, hidden_activity, _, _ = run_testing(
            dataset=dataset, model=rnn, n_trials=n_trials, verbose=False)
        epoch_result['accuracy'] = accuracy

        # task PCA
        pca_task, ve_task = fit_pca(hidden_activity, n_components=n_pc)
        epoch_result['rnn_task_ve_total'] = ve_task['total']

        # rest inference
        _, hidden_activity_rest, _ = run_testing_rest(
            rnn, smooth_noise=0, noise_mean=1, noise_sd=0.1,
            n_steps=fmri_rest_nsteps, fix_input_channels=[])

        # rest PCA
        pca_rest, ve_rest = fit_pca(hidden_activity_rest, n_components=n_pc)
        epoch_result['rnn_rest_ve_total'] = ve_rest['total']

        # fMRI projection
        if not skip_fmri:
            fmri_task_ve = np.zeros((n_pc, n_fmri_subj))
            fmri_rest_ve = np.zeros((n_pc, n_fmri_subj))
            for si in range(n_fmri_subj):
                ve_t = pca_projection(
                    fmri_task_ts[:, :, si], pca_task, centering_mean='data')
                fmri_task_ve[:, si] = ve_t['each']
                ve_r = pca_projection(
                    fmri_rest_ts[:, :, si], pca_rest, centering_mean='data')
                fmri_rest_ve[:, si] = ve_r['each']
            epoch_result['fmri_task_ve'] = fmri_task_ve
            epoch_result['fmri_rest_ve'] = fmri_rest_ve
        else:
            epoch_result['fmri_task_ve'] = None
            epoch_result['fmri_rest_ve'] = None

        run_result['epochs'][epoch] = epoch_result

    return run_idx, run_result


def compute_fmri_intersubj_baseline(fmri_ts, n_subj, n_pc):
    """Compute median inter-subject PCA variance explained (upper triangle)."""
    pca_list = []
    for si in range(n_subj):
        pca, _ = fit_pca(fmri_ts[:, :, si], n_components=n_pc)
        pca_list.append(pca)

    ve_matrix = np.zeros((n_subj, n_subj))
    for s1 in range(n_subj):
        for s2 in range(n_subj):
            ve = pca_projection(fmri_ts[:, :, s2], pca_list[s1], centering_mean='data')
            ve_matrix[s1, s2] = ve['total']

    triu_idx = np.triu_indices_from(ve_matrix, k=1)
    return np.median(ve_matrix[triu_idx])


# ---------------------------------------------------------------------------
# Path setup (mirrors notebook/existing scripts)
# ---------------------------------------------------------------------------

def get_paths(model_params_name):
    """Return datadir, modeldir, fmridir based on platform/user."""
    username = os.getenv('USER')
    if sys.platform == 'darwin':
        if username == 'ahmad':
            datadir = '/Users/ahmad/software/snaplab_github/neuro_rnn/data'
            modeldir = os.path.join('/Users/ahmad/data/rutgers/neuro_rnn/results/pytorch/model', model_params_name)
            # modeldir = os.path.join('/Volumes/Sabrent_2TB/rutgers/neuro_rnn/data', model_params_name)
            fmridir = '/Users/ahmad/data/rutgers/hcp'
    elif sys.platform == 'linux':
        if username == 'ab2792':
            datadir = '/home/ab2792/software/snaplab_github/neuro_rnn/data'
            modeldir = '/home/ab2792/data/neuro_rnn/results/pytorch/model'
            fmridir = '/home/ab2792/data/HCP/fmri'
        elif username == 'lindenmp':
            datadir = '/home/lindenmp/research_projects/neuro_rnn/data'
            modeldir = '/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/model_cpu'
            fmridir = '/media/lindenmp/storage_ssd/research_projects/neuro_rnn/data/fmri'
    return datadir, modeldir, fmridir


# ---------------------------------------------------------------------------
# Load spatial kernels
# ---------------------------------------------------------------------------

def load_kernels(datadir, hidden_size=100):
    """Load and normalize spatial kernel distance/similarity matrices."""
    eucl_kernel_file = os.path.join(datadir, 'schaefer200_centroids.csv')
    sa_kernel_file = os.path.join(datadir, 'schaefer200_sa-axis.npy')
    ut_kernel_file = os.path.join(datadir, 'schaefer200_ut-axis.npy')
    sf_kernel_file = os.path.join(datadir, 'schaefer200_cyto.npy')

    centroids = pd.read_csv(eucl_kernel_file)[:hidden_size]
    dist = distance.squareform(distance.pdist(centroids.set_index('ROI Name'), 'euclidean'))
    dist_eucl = utils.normalize_x(dist)

    sa_axis = np.load(sa_kernel_file)[:hidden_size]
    dist_sa = utils.normalize_x(utils.get_brainmap_distance(sa_axis))

    ut_axis = np.load(ut_kernel_file)[:hidden_size]
    dist_ut = utils.normalize_x(utils.get_brainmap_distance(ut_axis))

    sf_axis = np.load(sf_kernel_file)[:hidden_size]
    dist_sf = utils.normalize_x(utils.get_brainmap_distance(sf_axis))

    kernel_similarity_matrices = {
        'euclidean': 1 - dist_eucl,
        'sa_axis': 1 - dist_sa,
        'ut_axis': 1 - dist_ut,
        'sf_axis': 1 - dist_sf,
    }
    return kernel_similarity_matrices


# ---------------------------------------------------------------------------
# Load fMRI data
# ---------------------------------------------------------------------------

def load_fmri_data(datadir, fmridir, n_fmri_subj, hidden_size=100):
    """Load task and resting-state fMRI data for a common set of subjects."""
    tfmri_file = os.path.join(
        fmridir, 'hcpya_tfmri.pkl')
    fmri_data_file = os.path.join(
        fmridir, 'HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest.npy')
    fmri_data_df_file = os.path.join(
        fmridir, 'HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest_df.csv')

    n_nodes = hidden_size

    # task fMRI
    with open(tfmri_file, 'rb') as f:
        tfmri = pickle.load(f)
    fmri_task_key = 'tfMRIWMLR'
    fmri_task_parc = 'Schaefer2007'
    fmri_task_subjnames = list(tfmri.keys())

    # rest fMRI
    fmri_rest_data_df = pd.read_csv(fmri_data_df_file)
    use_rest = pd.concat([
        fmri_rest_data_df.rfMRI_available,
        fmri_rest_data_df.rfMRI_REST1_LR,
        fmri_rest_data_df.rfMRI_REST1_RL,
        fmri_rest_data_df.rfMRI_REST2_LR,
        fmri_rest_data_df.rfMRI_REST2_RL,
    ], axis=1).to_numpy().all(axis=1)
    fmri_rest_data_df = fmri_rest_data_df.iloc[use_rest]
    fmri_rest_subjnames = [str(x) for x in fmri_rest_data_df.Subject]
    fmri_rest_data_raw = np.load(fmri_data_file)[:, 16:116, 0, use_rest]

    # common subjects
    common = [s for s in fmri_task_subjnames if s in set(fmri_rest_subjnames)]
    rng = np.random.default_rng(seed=42)
    selected = [common[i] for i in rng.choice(len(common), size=n_fmri_subj, replace=False)]

    task_bool = [s in set(selected) for s in fmri_task_subjnames]
    rest_bool = [s in set(selected) for s in fmri_rest_subjnames]

    # assemble task fMRI array
    fmri_task_nsteps = min(
        tfmri[s][fmri_task_parc][fmri_task_key].shape[0] for s in selected)
    fmri_task_ts = np.zeros((fmri_task_nsteps, n_nodes, n_fmri_subj))
    for i, s in enumerate(selected):
        fmri_task_ts[:, :, i] = tfmri[s][fmri_task_parc][fmri_task_key][:fmri_task_nsteps, :n_nodes]

    # assemble rest fMRI array
    fmri_rest_ts = fmri_rest_data_raw[:, :, rest_bool]

    fmri_rest_nsteps = fmri_rest_ts.shape[0]

    print(f'fMRI subjects selected: {n_fmri_subj}')
    print(f'fMRI task shape: {fmri_task_ts.shape}')
    print(f'fMRI rest shape: {fmri_rest_ts.shape}')

    return fmri_task_ts, fmri_rest_ts, fmri_rest_nsteps, selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_params', required=True,
                        help='Model params CSV name (without .csv extension)')
    parser.add_argument('--rows', nargs='+', type=int, default=None,
                        help='Row indices to select from model params CSV')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='Number of evenly spaced epoch checkpoints to evaluate')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of test trials per run')
    parser.add_argument('--n_fmri_subj', type=int, default=100,
                        help='Number of fMRI subjects to use')
    parser.add_argument('--n_pc', type=int, default=5,
                        help='Number of PCA components')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for inference (-1 = all cores)')
    parser.add_argument('--max_runs', type=int, default=100,
                        help='Maximum number of runs per model')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (defaults to modeldir)')
    parser.add_argument('--skip_fmri', action='store_true',
                        help='Skip fMRI projection (compute only weight-kernel similarity + accuracy)')
    args = parser.parse_args()

    device = torch.device('cpu')

    # ---- Paths ----
    datadir, modeldir, fmridir = get_paths(args.model_params)
    outdir = args.outdir or modeldir
    os.makedirs(outdir, exist_ok=True)
    model_params_file = os.path.join(datadir, args.model_params + '.csv')

    print(f'Model params: {args.model_params}')
    print(f'Model dir:    {modeldir}')
    print(f'Output dir:   {outdir}')

    # ---- Load model params ----
    model_params, task_names, n_tasks, kernel_labels, n_kernels = \
        utils.get_params_dataframe(model_params_file, rows=args.rows or [], verbose=True)
    n_models = len(model_params)
    print(f'Models to process: {n_models}')

    # ---- Load kernels ----
    hidden_size = 100
    kernel_similarity_matrices = load_kernels(datadir, hidden_size)
    kernel_types_with_embedding = set(kernel_similarity_matrices.keys())

    # ---- Load fMRI data and compute baselines ----
    if not args.skip_fmri:
        fmri_task_ts, fmri_rest_ts, fmri_rest_nsteps, fmri_subjnames = \
            load_fmri_data(datadir, fmridir, args.n_fmri_subj, hidden_size)

        print('Computing fMRI inter-subject baselines...')
        t0 = time.time()
        fmri_task_baseline = compute_fmri_intersubj_baseline(
            fmri_task_ts, args.n_fmri_subj, args.n_pc)
        fmri_rest_baseline = compute_fmri_intersubj_baseline(
            fmri_rest_ts, args.n_fmri_subj, args.n_pc)
        print(f'  Task baseline: {fmri_task_baseline:.4f}')
        print(f'  Rest baseline: {fmri_rest_baseline:.4f}')
        print(f'  Time: {time.time() - t0:.1f}s')
    else:
        fmri_task_ts = fmri_rest_ts = None
        fmri_rest_nsteps = 1200
        fmri_task_baseline = fmri_rest_baseline = None
        fmri_subjnames = []

    # ---- Main loop ----
    all_results = []
    outfile = os.path.join(outdir, f'training_trajectory_{args.model_params}.pkl')

    def save_checkpoint():
        output = {
            'model_params_name': args.model_params,
            'model_params': model_params,
            'args': vars(args),
            'fmri_intersubj_baselines': {
                'task': fmri_task_baseline,
                'rest': fmri_rest_baseline,
            },
            'fmri_subjects': fmri_subjnames,
            'results': all_results,
        }
        with open(outfile, 'wb') as f:
            pickle.dump(output, f)

    for model_idx in range(n_models):
        this = model_params.iloc[model_idx]
        task_label = this.task_label
        kernel_label = this.kernel_label
        kernel_type = this.kernel_type
        print(f'\n{"="*70}')
        print(f'Model {model_idx+1}/{n_models}: {task_label} — {kernel_label}')
        print(f'{"="*70}')

        # get available epochs
        file_path_models = os.path.join(modeldir, this.file_str_models)
        if not os.path.isfile(file_path_models):
            print(f'  Model file not found, skipping: {this.file_str_models}')
            all_results.append(None)
            save_checkpoint()
            continue

        state_manager = ModelStateManager(file_path_models)
        n_runs_available, logged_epochs, _ = state_manager.get_info()
        n_runs = min(n_runs_available, args.max_runs)
        sampled_epochs = subsample_epochs(logged_epochs, args.n_epochs)
        print(f'  Runs: {n_runs}, Logged epochs: {len(logged_epochs)}, '
              f'Sampled: {len(sampled_epochs)} → {sampled_epochs}')

        # check if this model has a spatial kernel
        has_kernel = kernel_type in kernel_types_with_embedding
        kernel_sim = kernel_similarity_matrices.get(kernel_type, None)

        # remove delay variability for testing
        utils.check_if_supported(task=this.task_no_modifier, modifier=this.task_modifier)
        if 'delay' in this['env_kwargs']['timing']:
            d, _ = utils.parse_task_modifier(this.task_modifier)
            try:
                this['env_kwargs']['timing']['delay'] = d[0]
            except Exception:
                pass

        # ---- Pre-load all state dicts (sequential, lightweight) ----
        print(f'  Pre-loading state dicts ({n_runs} runs × {len(sampled_epochs)} epochs)...')
        t0 = time.time()
        all_state_dicts = {}  # {run_idx: {epoch: state_dict}}
        for run_idx in range(n_runs):
            all_state_dicts[run_idx] = {}
            for epoch in sampled_epochs:
                all_state_dicts[run_idx][epoch] = \
                    state_manager.load_model_states(run=run_idx, epoch=epoch)
        print(f'  Pre-loaded in {time.time() - t0:.1f}s')

        # ---- Create dataset and RNN shell once ----
        state_template = all_state_dicts[0][sampled_epochs[0]]
        dataset, rnn_template = create_dataset_and_rnn_shell(
            this, state_template, device)

        model_result = {
            'task': this.task_no_modifier,
            'task_label': task_label,
            'kernel_label': kernel_label,
            'kernel_type': kernel_type,
            'sampled_epochs': sampled_epochs,
            'n_runs': n_runs,
            'runs': [None] * n_runs,
        }

        # ---- Parallel inference across runs ----
        print(f'  Running inference (n_jobs={args.n_jobs})...')
        t0 = time.time()
        # get reg params for this model
        reg_type = this.reg_type if hasattr(this, 'reg_type') else 'l2'
        reg_weight = this.reg_weight if hasattr(this, 'reg_weight') else 0.0

        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_one_run)(
                run_idx,
                all_state_dicts[run_idx],
                sampled_epochs,
                this,
                rnn_template,
                dataset,
                has_kernel,
                kernel_sim,
                reg_type,
                reg_weight,
                args.n_trials,
                args.n_pc,
                fmri_rest_nsteps,
                fmri_task_ts,
                fmri_rest_ts,
                args.n_fmri_subj,
                args.skip_fmri,
                device,
            )
            for run_idx in range(n_runs)
        )
        for run_idx, run_result in results:
            model_result['runs'][run_idx] = run_result
        print(f'  Completed in {time.time() - t0:.1f}s')

        # free memory before next model
        del all_state_dicts
        all_results.append(model_result)

        # save after each model so partial results survive crashes
        save_checkpoint()
        print(f'  Saved checkpoint ({model_idx+1}/{n_models} models complete)')

    print(f'\nAll results saved to: {outfile}')


if __name__ == '__main__':
    main()
