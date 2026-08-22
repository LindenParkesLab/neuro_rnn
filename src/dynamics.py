"""Hidden-state dynamics and their correspondence with empirical fMRI.

Backs the ``biornn_analysis_dynamics`` notebook (paper Fig. 2 and Figs. S3, S4):
evaluating trained RNNs, summarising their hidden-state dynamics with PCA, and
measuring how much empirical fMRI variance those low-dimensional subspaces
capture relative to a random-subspace null.

Evaluation is **seeded** by default. A trained network is deterministic given
its inputs, but the task battery and the resting-state noise drive are drawn at
run time; leaving them unseeded makes every quantity downstream -- the PCs, the
variance explained, the z-scores -- vary between executions. The default
``EVAL_SEED`` matches the trajectory engine, so the two produce the same
hidden states for the same run.
"""

import os

import numpy as np
import scipy.stats as sp_stats
import torch
from scipy.spatial import distance

import src.pca_utils as pca_utils
import src.utils as utils
from src.config import get_paths
from src.neural_network import (ModelStateManager, create_rnn_and_env_for_model,
                                run_testing, run_testing_rest)
from src.performance import class_label

# Seed for the task battery and resting-state noise drive.
EVAL_SEED = 0

# Resting-state drive, matching the trajectory engine and the null machinery.
REST_NOISE_MEAN = 0.5
REST_NOISE_SD = 0.3


def load_model_table(model_params_name, rows, data_dir=None):
    """Return the params rows for the requested models, with paper class labels."""
    paths = get_paths(model_params_name)
    data_dir = data_dir or paths.data_dir
    params_file = f'{data_dir}/{model_params_name}.csv'
    table, *_ = utils.get_params_dataframe(params_file, rows=list(rows))
    table = table.copy()
    table['class_label'] = [class_label(r.mask_weights, r.kernel_type)
                            for _, r in table.iterrows()]
    return table


def resolve_epoch(model_info, model_dir, epoch):
    """Snap a requested epoch to the nearest saved checkpoint.

    Checkpoints are written every ``write_freq`` epochs and keyed 0-indexed, so
    the natural request "epoch 40,000" is stored as ``epoch_39999``. Callers
    should ask for the epoch they mean and let this resolve it.
    """
    manager = ModelStateManager(os.path.join(model_dir, model_info.file_str_models))
    _, logged_epochs, _ = manager.get_info()
    if epoch is None or epoch == -1:
        return logged_epochs[-1]
    return min(logged_epochs, key=lambda e: abs(e - epoch))


def evaluate_run(model_info, run, epoch, model_dir, device=None, n_trials=100,
                 rest_nsteps=1200, n_pc=5, seed=EVAL_SEED, return_hidden=False):
    """Drive one trained run under task and noise input, and summarise with PCA.

    Parameters
    ----------
    model_info : pandas Series
        One row of the params dataframe.
    run : int
        Which trained network of the ensemble.
    epoch : int
        Training checkpoint to evaluate at (the analyses use the convergence
        epoch derived in :mod:`src.performance`).
    model_dir : str
        Directory holding the trained ``*_models.h5`` files.
    device : torch.device, optional
        Defaults to CPU, which is what the published analyses used.
    n_trials : int
        Task trials in the test battery.
    rest_nsteps : int
        Timesteps of the resting-state (noise-driven) simulation. Match this to
        the empirical resting-state series length.
    n_pc : int
        Principal components retained per regime.
    seed : int or None
        Seeds the task battery and the noise drive. ``None`` draws both afresh,
        so results will differ between calls -- see the module docstring.
    return_hidden : bool
        Also return the raw hidden-state trajectories, e.g. to compute the
        network's own functional connectivity.

    Returns
    -------
    dict
        ``accuracy``, ``pca_task``/``pca_rest`` (fitted PCA objects),
        ``ve_task``/``ve_rest`` (explained-variance dicts over the RNN's own
        activity, used for Fig. S3), and ``n_pc``.
    """
    device = device or torch.device('cpu')
    torch.set_num_threads(1)

    epoch = resolve_epoch(model_info, model_dir, epoch)
    dataset, rnn = create_rnn_and_env_for_model(model_info, run, epoch, model_dir, device)

    accuracy, _, _, hidden_task, _, _ = run_testing(
        dataset=dataset, model=rnn, n_trials=n_trials, verbose=False, test_seed=seed)
    _, hidden_rest, _ = run_testing_rest(
        rnn, smooth_noise=0, noise_mean=REST_NOISE_MEAN, noise_sd=REST_NOISE_SD,
        n_steps=rest_nsteps, fix_input_channels=[], seed=seed)

    pca_task, ve_task = pca_utils.fit_pca(hidden_task, n_pc, return_variance=True)
    pca_rest, ve_rest = pca_utils.fit_pca(hidden_rest, n_pc, return_variance=True)

    out = {'accuracy': accuracy, 'n_pc': n_pc,
           'pca_task': pca_task, 'pca_rest': pca_rest,
           've_task': ve_task, 've_rest': ve_rest}
    if return_hidden:
        out['hidden_task'] = hidden_task
        out['hidden_rest'] = hidden_rest
    return out


def fmri_variance_explained(result, fmri_task_ts, fmri_rest_ts):
    """Subject-averaged fMRI variance captured by a run's PC subspaces.

    Each regime is matched to its counterpart modality: the task-driven
    subspace is tested against task fMRI, the noise-driven subspace against
    resting-state fMRI.
    """
    return {
        'task': pca_utils.subspace_ve_all(fmri_task_ts, result['pca_task']),
        'rest': pca_utils.subspace_ve_all(fmri_rest_ts, result['pca_rest']),
    }


def evaluate_model(model_info, epoch, fmri_task_ts, fmri_rest_ts, model_dir,
                   runs=None, n_pc=5, n_trials=100, seed=EVAL_SEED,
                   device=None, progress=None):
    """Evaluate every run of one model and collect its dynamics measures.

    Parameters
    ----------
    model_info : pandas Series
        One row of the params dataframe.
    epoch : int
        Checkpoint to evaluate at.
    fmri_task_ts, fmri_rest_ts : (time, nodes, subjects) arrays
    model_dir : str
    runs : sequence of int, optional
        Defaults to every run in the ensemble.
    progress : callable, optional
        Wraps the run iterator, e.g. ``tqdm``.

    Returns
    -------
    dict
        ``runs`` (indices evaluated), ``accuracy`` (n_runs,),
        ``ve_task``/``ve_rest`` (n_runs,) subject-averaged fMRI variance
        explained, ``components_task``/``components_rest``
        (n_runs, n_pc, n_nodes) PC loadings, and ``explained_variance_task``
        (n_runs, n_pc) the RNN's own variance per PC.
    """
    if runs is None:
        runs = range(int(model_info.n_runs))
    runs = list(runs)
    iterator = progress(runs) if progress is not None else runs

    out = {k: [] for k in ('accuracy', 've_task', 've_rest',
                           'components_task', 'components_rest',
                           'explained_variance_task')}
    for run in iterator:
        result = evaluate_run(model_info, run, epoch, model_dir, device=device,
                              n_trials=n_trials, rest_nsteps=fmri_rest_ts.shape[0],
                              n_pc=n_pc, seed=seed)
        ve = fmri_variance_explained(result, fmri_task_ts, fmri_rest_ts)
        out['accuracy'].append(result['accuracy'])
        out['ve_task'].append(ve['task'])
        out['ve_rest'].append(ve['rest'])
        out['components_task'].append(result['pca_task'].components_)
        out['components_rest'].append(result['pca_rest'].components_)
        out['explained_variance_task'].append(result['ve_task']['each'])

    return {'runs': np.asarray(runs, int),
            **{k: np.asarray(v) for k, v in out.items()}}


def group_components(components, n_components=5):
    """Sign-aligned mean PC loadings across runs, for surface plotting.

    PCA fixes each component only up to sign, so a plain average across
    independently fitted runs cancels real structure. See
    :func:`src.pca_utils.align_and_average_components`.
    """
    return pca_utils.align_and_average_components(
        [np.asarray(c) for c in components], n_components=n_components)


def learned_runs(accuracy, chance=1.0 / 3.0):
    """Boolean mask of runs that learned the task, for excluding dead runs.

    ``accuracy`` is a fraction, matching ``run_testing``'s output.
    """
    return np.asarray(accuracy, float) > chance


# ---------------------------------------------------------------------------
# Weight-geometry correspondence (Fig. S4)
# ---------------------------------------------------------------------------

def load_hidden_weights(model_info, run, epoch, model_dir):
    """Recurrent weight matrix ``W_hh`` of one run at a checkpoint."""
    epoch = resolve_epoch(model_info, model_dir, epoch)
    manager = ModelStateManager(os.path.join(model_dir, model_info.file_str_models))
    state = manager.load_model_states(run, epoch)
    return np.asarray(state['rnn.weight_hh_l0'])


def weight_kernel_similarity(weights, kernel, metric='cosine', use_abs=False,
                             center=True):
    """Similarity between a recurrent weight matrix and a spatial kernel.

    Off-diagonal entries only -- self-connections carry no spatial information.

    Parameters
    ----------
    weights : (n, n) array
        Recurrent weights.
    kernel : (n, n) array
        Spatial **similarity** matrix (proximity), as returned by
        :func:`src.fmri_io.load_kernels`.
    metric : {'cosine', 'spearman', 'pearson'}
    use_abs : bool
        Compare connection *magnitude* rather than signed weight, leaving
        excitatory/inhibitory sign free.
    center : bool
        Mean-center both vectors before a cosine. **On by default**, and that
        matters: the kernel is an all-positive similarity, so an uncentred
        cosine is dominated by the shared offset rather than by how the weights
        actually track proximity. Centred, it measures signed alignment (and is
        then equal to a Pearson correlation, and invariant to how the kernel was
        rescaled). Rank-based metrics are unaffected either way.
    """
    off_diagonal = ~np.eye(weights.shape[0], dtype=bool)
    w = np.abs(weights[off_diagonal]) if use_abs else weights[off_diagonal]
    k = kernel[off_diagonal]

    if metric == 'spearman':
        return float(sp_stats.spearmanr(w, k).statistic)
    if metric == 'pearson':
        return float(sp_stats.pearsonr(w, k).statistic)
    if metric != 'cosine':
        raise ValueError(f"unknown metric {metric!r}")
    if center:
        w = w - w.mean()
        k = k - k.mean()
    return float(np.dot(w, k) / (np.linalg.norm(w) * np.linalg.norm(k)))


def subject_geometry_kernels(coords_file, subjects, hidden_size=100,
                             normalization='mean'):
    """Per-subject spatial similarity matrices from individual parcel centroids.

    Mirrors the group kernel used for training (normalized distance, then
    ``1 - distance``), but built from each subject's own coordinates.

    Parameters
    ----------
    coords_file : str
        ``.npz`` keyed by subject ID, each holding ``(n_parcels, 3)`` centroids.
    subjects : sequence of str
        Subject IDs, in the order the fMRI arrays use.
    hidden_size : int
        Parcels to keep (100 = left hemisphere of the Schaefer-200 atlas).

    Returns
    -------
    (n_subjects, hidden_size, hidden_size) array of similarity matrices.
    """
    coords = np.load(coords_file, allow_pickle=True)
    out = []
    for subject in subjects:
        key = str(subject)
        if key not in coords:
            raise KeyError(f'subject {key} not found in {os.path.basename(coords_file)}')
        xyz = np.asarray(coords[key])[:hidden_size]
        dist = distance.squareform(distance.pdist(xyz, 'euclidean'))
        out.append(1.0 - utils.normalize_x(dist, normalization))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Task-evoked vs intrinsic dynamics (Fig. 3b)
# ---------------------------------------------------------------------------

def variance_decomposition(result, fmri_ts):
    """Split the fMRI variance a run explains into unique and shared parts.

    The same network is driven two ways: by the task, and by unstructured noise.
    Each regime yields a subspace, and together they span a pooled basis. The
    variance one regime explains that the other cannot is its *unique*
    contribution; what both account for is *shared*, and is attributable to the
    architecture rather than to either input.

    Component names match those of
    :func:`src.null_utils.random_subspace_null` with ``paired=True``, so a value
    can be calibrated against the null of the same component.

    Parameters
    ----------
    result : dict
        Output of :func:`evaluate_run`, holding both fitted subspaces.
    fmri_ts : (time, nodes, subjects) array
        The empirical modality to explain.

    Returns
    -------
    dict
        ``ve_a`` (task-driven), ``ve_b`` (noise-driven), ``ve_union``,
        ``unique_a``, ``unique_b``, ``shared``, and ``union_rank`` -- normally
        ``2 * n_pc``, lower if the two subspaces overlap.
    """
    basis_task = result['pca_task'].components_
    basis_noise = result['pca_rest'].components_
    basis_union = pca_utils.orth_basis(np.vstack([basis_task, basis_noise]))

    ve_a = pca_utils.subspace_ve_all(fmri_ts, basis_task)
    ve_b = pca_utils.subspace_ve_all(fmri_ts, basis_noise)
    ve_union = pca_utils.subspace_ve_all(fmri_ts, basis_union)

    return {'ve_a': ve_a, 've_b': ve_b, 've_union': ve_union,
            'unique_a': ve_union - ve_b,
            'unique_b': ve_union - ve_a,
            'shared': ve_a + ve_b - ve_union,
            'union_rank': basis_union.shape[0]}


def model_functional_connectivity(hidden_activity, nodes=None):
    """Functional connectivity of a network's own hidden activity.

    Concatenates trials into one time series and correlates nodes, giving the
    model-side counterpart of an empirical FC matrix.

    Parameters
    ----------
    hidden_activity : list of (time, nodes) arrays, or one array
        Hidden-state trajectories, as returned by
        :func:`evaluate_run` with ``return_hidden=True``.
    nodes : array-like of int, optional
        Restrict to a subset, e.g. the bystander nodes.
    """
    activity = pca_utils.stack_trials(hidden_activity)
    if nodes is not None:
        activity = activity[:, np.asarray(nodes, int)]
    return utils.compute_fc(activity)
