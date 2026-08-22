"""Compute weight–kernel similarity, fMRI variance explained, and network topology
across training epochs.

Iterates over models × runs × sampled epochs, computes metrics at each checkpoint,
and saves results as a pickle file for downstream plotting.

Topology metrics (segregation + centrality/hubs) are computed twice per checkpoint:
on the whole-network recurrent weight matrix ('topology') and on the bystander/
intermediate nodes only ('topology_bystander', excluding input and output nodes),
each anchored against the spatial kernel and group task/rest fMRI functional
connectivity (reference metric vectors + per-epoch matrix-level similarity). Select
a single kernel type (e.g. Euclidean bioRNNs) via --rows; the code itself is
kernel-type-agnostic.

Usage:
    python scripts/biornn_results_dynamics_trajectory.py \
        --model_params model_params_202606d \
        --rows 2 \
        --n_epochs 10 \
        --max_epoch 30000 \
        --n_trials 100 \
        --n_fmri_subj 100 \
        --n_pc 5 \
        --n_jobs -1 \
        --rc_perms 1000 \
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
from sklearn.decomposition import PCA
from joblib import Parallel, delayed
import torch

import src.utils as utils
from src.fmri_io import get_paths, load_kernels, load_fmri_data
from src.topology import compute_topology_metrics, compute_reference_similarity
from src.neural_network import (
    ModelStateManager,
    RNN,
    run_testing,
    run_testing_rest,
)
import neurogym as ngym
import torch.nn as nn

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def subsample_epochs(logged_epochs, n_epochs):
    """Select n_epochs evenly spaced checkpoints, always including first and last.

    If n_epochs is None or -1, all logged epochs are returned.
    """
    if n_epochs is None or n_epochs == -1 or n_epochs >= len(logged_epochs):
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
    """Compute similarity metrics between learned weights and spatial kernel.

    Both signed and magnitude-based (``*_abs``) variants are returned. The signed
    metrics align the raw weight (sign = E/I) with the kernel; the ``*_abs``
    metrics align connection *magnitude* |W| with the kernel, leaving sign free.
    The ``*_abs`` variants give a common, sign-free readout for comparing models
    trained with different objectives (e.g. ``pearson_l2s`` vs ``pearson_abs_l2s``).
    """
    include_mask = ~np.eye(W_hh.shape[0], dtype=np.bool_)
    w_vec = W_hh[include_mask]
    k_vec = kernel_similarity[include_mask]
    w_abs = np.abs(w_vec)

    r_spearman, _ = stats.spearmanr(w_vec, k_vec)
    r_spearman_abs, _ = stats.spearmanr(w_abs, k_vec)
    eucl_dist = np.linalg.norm(w_vec - k_vec)
    cos_sim = np.dot(w_vec, k_vec) / (np.linalg.norm(w_vec) * np.linalg.norm(k_vec))
    cos_sim_abs = np.dot(w_abs, k_vec) / (np.linalg.norm(w_abs) * np.linalg.norm(k_vec))

    return {
        'spearman': r_spearman,
        'spearman_abs': r_spearman_abs,
        'euclidean': eucl_dist,
        'cosine': cos_sim,
        'cosine_abs': cos_sim_abs,
    }


def compute_losses(rnn, dataset, n_trials, reg_type, reg_weight):
    """Compute loss components at a checkpoint.

    Returns a dict with:
        task_loss: cross-entropy loss (mean over trials)
        l2_component: reg_weight * sum(W_hh^2) / hidden_size (only for *_l2s)
        pearson_component: reg_weight * (1 - corr) (only for pearson* types)
        total_spatial_loss: reg_weight * regularization(W_hh, ...)
        total_loss: task_loss + total_spatial_loss
    """

    W_hh = rnn.rnn.weight_hh_l0
    hidden_size = W_hh.shape[0]
    kernel = rnn.regularization_kernel  # may be None

    # task loss: run each trial through the model. The RNN is sequence-first
    # (batch_first=False), so a single trial must be shaped (seq_len, batch=1,
    # input); env.ob is (seq_len, input). Using env.ob[np.newaxis, :, :] shaped it
    # (1, seq_len, input), which the RNN read as seq_len=1 over a batch of
    # timesteps -- i.e. no recurrence across time. On a memory task that yields a
    # meaningless cross-entropy that spuriously *rises* over training; use the same
    # (seq_len, 1, input) convention as run_testing so this matches training CE.
    criterion = nn.CrossEntropyLoss()
    task_losses = []
    env = dataset.env
    env.reset(seed=0)
    for _ in range(n_trials):
        env.new_trial()
        obs = torch.tensor(env.ob[:, np.newaxis, :], dtype=torch.float32,
                           device=W_hh.device)
        gt = torch.tensor(env.gt, dtype=torch.long, device=W_hh.device)
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
        if reg_weight == 0 or (kernel is None and reg_type in ('pearson', 'pearson_l2s', 'pearson_abs', 'pearson_abs_l2s')):
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

        # decompose for pearson_l2s / pearson_abs_l2s
        if reg_type in ('pearson_l2s', 'pearson_abs_l2s') and kernel is not None:
            # L2 component: reg_weight * sum(W^2) / hidden_size (unweighted by matrix)
            l2_val = reg_weight * torch.square(W_hh).sum() / hidden_size
            result['l2_component'] = l2_val.item()

            # Pearson component: reg_weight * (1 - corr); abs variant correlates |W|
            similarity = 1.0 - kernel
            off_diag = ~torch.eye(hidden_size, dtype=torch.bool, device=W_hh.device)
            w_flat = W_hh[off_diag]
            if reg_type == 'pearson_abs_l2s':
                w_flat = torch.abs(w_flat)
            s_flat = similarity[off_diag]
            corr = torch.corrcoef(torch.stack([w_flat, s_flat]))[0, 1]
            pearson_val = reg_weight * (1.0 - corr)
            result['pearson_component'] = pearson_val.item()

        elif reg_type in ('pearson', 'pearson_abs') and kernel is not None:
            result['pearson_component'] = result['total_spatial_loss']

        result['total_loss'] = task_loss + result['total_spatial_loss']

    return result


def get_node_subsets(state_dict, hidden_size):
    """Node subsets used for VE: the full network and the bystander nodes only.

    Returns a dict mapping subset name → array of node indices. Always includes
    'all'. If I/O masks are present, also includes 'bystander' (intermediate
    nodes: neither input nor output).
    """
    all_idx = np.arange(hidden_size)
    subsets = {'all': all_idx}

    if 'input_weight_mask' not in state_dict:
        return subsets

    iw = state_dict['input_weight_mask']
    if hasattr(iw, 'numpy'):
        iw = iw.numpy()
    input_mask = iw[:, 0].astype(bool) if iw.ndim == 2 else iw.astype(bool)

    ow = state_dict['output_weight_mask']
    if hasattr(ow, 'numpy'):
        ow = ow.numpy()
    output_mask = ow[0, :].astype(bool) if ow.ndim == 2 else ow.astype(bool)

    bystander_mask = ~(input_mask | output_mask)
    subsets['bystander'] = np.where(bystander_mask)[0]

    return subsets


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


def _topology_for_subset(W_hh, idx, ref_matrices, rc_perms):
    """Topology metrics + reference similarity for a node subset.

    `idx=None` uses the full network; otherwise the recurrent matrix and each
    reference matrix are restricted to the `idx` nodes before computing metrics.
    """
    if idx is None:
        W_sub = W_hh
        slice_ref = lambda mat: mat
    else:
        ix = np.ix_(idx, idx)
        W_sub = W_hh[ix]
        slice_ref = lambda mat: mat[ix]
    return {
        'metrics': compute_topology_metrics(W_sub, rc_perms=rc_perms),
        'reference_similarity': {
            name: (compute_reference_similarity(W_sub, slice_ref(mat))
                   if mat is not None else None)
            for name, mat in ref_matrices.items()
        },
    }


def _bystander_node_idx(node_subsets):
    """Intermediate (bystander) node indices: neither input nor output.

    Returns an empty array if I/O masks were unavailable (only 'all' present).
    """
    by = node_subsets.get('bystander')
    return np.asarray(by, dtype=int) if by is not None else np.array([], dtype=int)


def process_one_run(run_idx, epoch_states, sampled_epochs, model_info,
                    rnn_template, dataset, has_kernel, kernel_sim,
                    reg_type, reg_weight,
                    n_trials, n_pc, fmri_rest_nsteps,
                    fmri_task_ts, fmri_rest_ts, n_fmri_subj, skip_fmri,
                    node_subsets, compute_topology, topo_subsets, rc_perms, ref_matrices,
                    device, test_seed=0):
    """Process all epochs for a single run. Designed to run in a worker process."""
    torch.set_num_threads(1)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    rnn = copy.deepcopy(rnn_template)
    run_result = {'epochs': {}}
    # fixed eval battery (identical task test-trials + resting-state noise for every
    # epoch/run/model) when test_seed >= 0 -> deterministic, comparable trajectories;
    # a negative test_seed keeps the legacy stochastic evaluation.
    eval_seed = None if (test_seed is None or test_seed < 0) else int(test_seed)

    for epoch in sampled_epochs:
        state = epoch_states[epoch]
        rnn.load_state_dict(state)
        rnn.eval()

        epoch_result = {}

        # raw recurrent weights for this checkpoint (used below)
        W_hh = rnn.rnn.weight_hh_l0.detach().cpu().numpy()

        # weight-kernel similarity
        if has_kernel:
            epoch_result['weight_kernel_similarity'] = \
                compute_weight_kernel_similarity(W_hh, kernel_sim)
        else:
            epoch_result['weight_kernel_similarity'] = None

        # network topology + reference anchoring, computed twice: for the full
        # network ('topology') and for the bystander/intermediate nodes only
        # ('topology_bystander', excluding input and output nodes).
        if compute_topology:
            do_full = topo_subsets in ('both', 'all')
            do_by = topo_subsets in ('both', 'bystanders')
            epoch_result['topology'] = (
                _topology_for_subset(W_hh, None, ref_matrices, rc_perms)
                if do_full else None)
            bystander_idx = _bystander_node_idx(node_subsets)
            epoch_result['topology_bystander'] = (
                _topology_for_subset(W_hh, bystander_idx, ref_matrices, rc_perms)
                if (do_by and bystander_idx.size > 2) else None)
        else:
            epoch_result['topology'] = None
            epoch_result['topology_bystander'] = None

        # losses
        epoch_result['losses'] = compute_losses(
            rnn, dataset, n_trials, reg_type, reg_weight)

        # task inference
        accuracy, _, _, hidden_activity, _, _ = run_testing(
            dataset=dataset, model=rnn, n_trials=n_trials, verbose=False, test_seed=eval_seed)
        epoch_result['accuracy'] = accuracy

        # rest inference (needed for all subsets below)
        _, hidden_activity_rest, _ = run_testing_rest(
            rnn, smooth_noise=0, noise_mean=0.5, noise_sd=0.3,
            n_steps=fmri_rest_nsteps, fix_input_channels=[], seed=eval_seed)

        # PCA + fMRI projection for each node subset
        epoch_result['node_subsets'] = {}
        for subset_name, node_idx in node_subsets.items():
            sub = {}

            # slice hidden activity to this subset of nodes
            ha_task_sub = [arr[:, node_idx] for arr in hidden_activity]
            ha_rest_sub = hidden_activity_rest[:, node_idx]

            # task PCA
            pca_task, ve_task = fit_pca(ha_task_sub, n_components=n_pc)
            sub['rnn_task_ve_total'] = ve_task['total']

            # rest PCA
            pca_rest, ve_rest = fit_pca(ha_rest_sub, n_components=n_pc)
            sub['rnn_rest_ve_total'] = ve_rest['total']

            # fMRI projection (same node indices select matching parcels)
            if not skip_fmri:
                fmri_task_ve = np.zeros((n_pc, n_fmri_subj))
                fmri_rest_ve = np.zeros((n_pc, n_fmri_subj))
                for si in range(n_fmri_subj):
                    ve_t = pca_projection(
                        fmri_task_ts[:, node_idx, si], pca_task,
                        centering_mean='data')
                    fmri_task_ve[:, si] = ve_t['each']
                    ve_r = pca_projection(
                        fmri_rest_ts[:, node_idx, si], pca_rest,
                        centering_mean='data')
                    fmri_rest_ve[:, si] = ve_r['each']
                sub['fmri_task_ve'] = fmri_task_ve
                sub['fmri_rest_ve'] = fmri_rest_ve
            else:
                sub['fmri_task_ve'] = None
                sub['fmri_rest_ve'] = None

            epoch_result['node_subsets'][subset_name] = sub

        # keep top-level keys for backward compatibility ('all' subset)
        all_sub = epoch_result['node_subsets']['all']
        epoch_result['rnn_task_ve_total'] = all_sub['rnn_task_ve_total']
        epoch_result['rnn_rest_ve_total'] = all_sub['rnn_rest_ve_total']
        epoch_result['fmri_task_ve'] = all_sub['fmri_task_ve']
        epoch_result['fmri_rest_ve'] = all_sub['fmri_rest_ve']

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_params', required=True,
                        help='Model params CSV name (without .csv extension)')
    parser.add_argument('--rows', nargs='+', type=int, default=None,
                        help='Row indices to select from model params CSV')
    parser.add_argument('--n_epochs', type=int, default=None,
                        help='Number of evenly spaced epoch checkpoints to evaluate '
                             '(default: None = use all available; -1 also means all)')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of test trials per run')
    parser.add_argument('--n_fmri_subj', type=int, default=100,
                        help='Number of fMRI subjects to use')
    parser.add_argument('--n_pc', type=int, default=5,
                        help='Number of PCA components')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for inference (-1 = all cores)')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Only analyse epochs up to this value (inclusive)')
    parser.add_argument('--max_runs', type=int, default=100,
                        help='Maximum number of runs per model')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (defaults to model_dir)')
    parser.add_argument('--skip_fmri', action='store_true',
                        help='Skip fMRI projection (compute only weight-kernel similarity + accuracy)')
    parser.add_argument('--skip_topology', action='store_true',
                        help='Skip network-topology metrics (whole-network graph analysis)')
    parser.add_argument('--node_subsets', choices=['both', 'bystanders', 'all'],
                        default='both',
                        help="Which node subset(s) to compute TOPOLOGY for: 'both' (full "
                             "network + bystanders; default), 'bystanders' (skip full-network "
                             "topology, faster), or 'all' (full network only). fMRI VE is "
                             "always computed for all subsets regardless of this flag.")
    parser.add_argument('--rc_perms', type=int, default=1000,
                        help='Permutations for normalized rich-club (0 = raw, no normalization)')
    parser.add_argument('--test_seed', type=int, default=0,
                        help='Fixed seed for the eval battery (task test-trials + resting-state '
                             'noise), so trajectories are deterministic and comparable across '
                             'epochs/runs/models. Use a negative value for legacy stochastic eval.')
    args = parser.parse_args()

    device = torch.device('cpu')

    # ---- Paths ----
    paths = get_paths(args.model_params, require='all')
    outdir = args.outdir or paths.model_dir
    os.makedirs(outdir, exist_ok=True)
    model_params_file = os.path.join(paths.data_dir, args.model_params + '.csv')

    print(f'Model params: {args.model_params}')
    print(f'Model dir:    {paths.model_dir}')
    print(f'Output dir:   {outdir}')

    # ---- Load model params ----
    model_params, task_names, n_tasks, kernel_labels, n_kernels = \
        utils.get_params_dataframe(model_params_file, rows=args.rows or [], verbose=True)
    n_models = len(model_params)
    print(f'Models to process: {n_models}')

    # ---- Load kernels ----
    hidden_size = 100
    kernel_similarity_matrices = load_kernels(paths.data_dir, hidden_size)
    kernel_types_with_embedding = set(kernel_similarity_matrices.keys())

    # ---- Load fMRI data and compute baselines ----
    if not args.skip_fmri:
        fmri_task_ts, fmri_rest_ts, fmri_rest_nsteps, fmri_subjnames = \
            load_fmri_data(paths.data_dir, paths.fmri_dir, args.n_fmri_subj, hidden_size)

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

    # ---- Topology reference matrices (group fMRI FC, computed once) ----
    compute_topology = not args.skip_topology

    def group_fc(ts):
        """Subject-averaged functional connectivity from (time, node, subj)."""
        fcs = [utils.compute_fc(ts[:, :, si]) for si in range(ts.shape[2])]
        return np.nanmean(np.stack(fcs, axis=0), axis=0)

    fmri_task_fc = fmri_rest_fc = None
    ref_topo_fmri_task = ref_topo_fmri_rest = None
    if compute_topology and not args.skip_fmri:
        print('Computing group fMRI FC + reference topology...')
        t0 = time.time()
        fmri_task_fc = group_fc(fmri_task_ts)
        fmri_rest_fc = group_fc(fmri_rest_ts)
        # full-network reference topology only needed when computing full-network topology
        if args.node_subsets in ('both', 'all'):
            ref_topo_fmri_task = compute_topology_metrics(fmri_task_fc, rc_perms=args.rc_perms)
            ref_topo_fmri_rest = compute_topology_metrics(fmri_rest_fc, rc_perms=args.rc_perms)
        print(f'  Time: {time.time() - t0:.1f}s')

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

    # cache bystander inter-subject baselines keyed by the bystander node set
    # (avoids recomputation when models share the same I/O layout)
    bystander_baseline_cache = {}

    for model_idx in range(n_models):
        this = model_params.iloc[model_idx]

        # skip spatial-null rows: their trained files are the '-null{p}' perm
        # ensembles (offset / spanning run indices), not this base row -- the
        # trajectory analysis is for the primary models only.
        if int(getattr(this, 'null_perms', 0) or 0) > 0:
            print(f'\nModel {model_idx+1}/{n_models}: null-ensemble row '
                  f'(null_perms={int(this.null_perms)}) — skipping.')
            all_results.append(None)
            save_checkpoint()
            continue

        task_label = this.task_label
        kernel_label = this.kernel_label
        kernel_type = this.kernel_type
        print(f'\n{"="*70}')
        print(f'Model {model_idx+1}/{n_models}: {task_label} — {kernel_label}')
        print(f'{"="*70}')

        # get available epochs
        file_path_models = os.path.join(paths.model_dir, this.file_str_models)
        if not os.path.isfile(file_path_models):
            print(f'  Model file not found, skipping: {this.file_str_models}')
            all_results.append(None)
            save_checkpoint()
            continue

        state_manager = ModelStateManager(file_path_models)
        n_runs_available, logged_epochs, _ = state_manager.get_info()
        if args.max_epoch is not None:
            logged_epochs = [e for e in logged_epochs if e <= args.max_epoch]
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

        # ---- Determine node subsets for VE (full network + bystanders only) ----
        node_subsets = get_node_subsets(state_template, hidden_size)
        if len(node_subsets) > 1:
            print(f'  Node subsets: {", ".join(f"{k} ({len(v)} nodes)" for k, v in node_subsets.items())}')
        else:
            print(f'  Node subsets: all ({hidden_size} nodes, no I/O masks)')

        # ---- Bystander-only inter-subject fMRI baseline (region-matched) ----
        fmri_baselines_bystander = None
        if not args.skip_fmri:
            by_idx = _bystander_node_idx(node_subsets)
            if by_idx.size > 2:
                bkey = tuple(int(i) for i in by_idx)
                if bkey not in bystander_baseline_cache:
                    bystander_baseline_cache[bkey] = {
                        'task': compute_fmri_intersubj_baseline(
                            fmri_task_ts[:, by_idx, :], args.n_fmri_subj, args.n_pc),
                        'rest': compute_fmri_intersubj_baseline(
                            fmri_rest_ts[:, by_idx, :], args.n_fmri_subj, args.n_pc),
                    }
                fmri_baselines_bystander = bystander_baseline_cache[bkey]

        # ---- Topology references for this model (kernel + group fMRI FC) ----
        ref_matrices = {'kernel': None, 'fmri_task': None, 'fmri_rest': None}
        reference_topology = None
        reference_topology_bystander = None
        if compute_topology:
            kernel_ref = kernel_sim if has_kernel else None
            ref_matrices = {
                'kernel': kernel_ref,
                'fmri_task': fmri_task_fc,
                'fmri_rest': fmri_rest_fc,
            }
            if args.node_subsets in ('both', 'all'):
                reference_topology = {
                    'kernel': (compute_topology_metrics(kernel_ref, rc_perms=args.rc_perms)
                               if kernel_ref is not None else None),
                    'fmri_task': ref_topo_fmri_task,
                    'fmri_rest': ref_topo_fmri_rest,
                }
            # bystander-only topology of the reference networks (for overlays on
            # the bystander RNN-topology plots)
            if args.node_subsets in ('both', 'bystanders'):
                by_idx = _bystander_node_idx(node_subsets)
                if by_idx.size > 2:
                    bix = np.ix_(by_idx, by_idx)
                    reference_topology_bystander = {
                        name: (compute_topology_metrics(mat[bix], rc_perms=args.rc_perms)
                               if mat is not None else None)
                        for name, mat in ref_matrices.items()
                    }

        model_result = {
            'task': this.task_no_modifier,
            'task_label': task_label,
            'kernel_label': kernel_label,
            'kernel_type': kernel_type,
            'sampled_epochs': sampled_epochs,
            'n_runs': n_runs,
            'node_subsets': {k: v.tolist() for k, v in node_subsets.items()},
            'reference_topology': reference_topology,
            'reference_topology_bystander': reference_topology_bystander,
            'fmri_intersubj_baselines_bystander': fmri_baselines_bystander,
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
                node_subsets,
                compute_topology,
                args.node_subsets,
                args.rc_perms,
                ref_matrices,
                device,
                args.test_seed,
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
