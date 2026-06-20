"""Shared fMRI loading and preprocessing utilities.

Single source of truth for the HCP task/rest fMRI pipeline and spatial kernels
used by the training-trajectory engine (``biornn_results_dynamics_trajectory.py``)
and the results notebooks (``biornn_results_dynamics*.ipynb``): host path
resolution, seeded common-subject selection, global signal regression, spatial
kernel loading, and the full task/rest loader. These were previously duplicated
(and drifting) across all three, which is what made the GSR/selection logic hard
to keep aligned -- everything now lives here.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.spatial import distance

import src.utils as utils


# ---------------------------------------------------------------------------
# Path setup
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
# fMRI preprocessing
# ---------------------------------------------------------------------------

def global_signal_regression(ts):
    """Regress the global signal out of a (time, nodes, subjects) array in place.

    For each subject, the global signal (mean across nodes per timepoint) is
    removed from every node's time series via OLS. Returns ``ts``.
    """
    for si in range(ts.shape[2]):
        gs = ts[:, :, si].mean(axis=1, keepdims=True)   # (time, 1)
        beta = (gs.T @ ts[:, :, si]) / (gs.T @ gs)      # (1, nodes)
        ts[:, :, si] -= gs @ beta
    return ts


def select_common_subjects(task_subjnames, rest_subjnames, n_fmri_subj,
                           seed=42, pick_random=True):
    """Select ``n_fmri_subj`` subjects present in both the task and rest sets.

    Filters the task subject list to those also in the rest set, then draws
    ``n_fmri_subj`` of them. With ``pick_random`` (default) the draw is seeded
    (``np.random.default_rng(seed)``) so every caller selects the identical
    subject set -- required for the trajectory notebook's chance normalization
    to align per subject with the VE arrays computed elsewhere.
    """
    common = [s for s in task_subjnames if s in set(rest_subjnames)]
    if pick_random:
        rng = np.random.default_rng(seed=seed)
        return [common[i] for i in rng.choice(len(common), size=n_fmri_subj, replace=False)]
    return common[:n_fmri_subj]


# ---------------------------------------------------------------------------
# Spatial kernels
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
# Task + rest fMRI loader
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

    # common subjects (seeded so every caller selects the identical set)
    selected = select_common_subjects(
        fmri_task_subjnames, fmri_rest_subjnames, n_fmri_subj, seed=42)
    rest_bool = [s in set(selected) for s in fmri_rest_subjnames]

    # assemble task fMRI array (over the selected subjects)
    fmri_task_nsteps = min(
        tfmri[s][fmri_task_parc][fmri_task_key].shape[0] for s in selected)
    fmri_task_ts = np.zeros((fmri_task_nsteps, n_nodes, n_fmri_subj))
    for i, s in enumerate(selected):
        fmri_task_ts[:, :, i] = tfmri[s][fmri_task_parc][fmri_task_key][:fmri_task_nsteps, :n_nodes]

    # assemble rest fMRI array
    fmri_rest_ts = fmri_rest_data_raw[:, :, rest_bool]
    fmri_rest_nsteps = fmri_rest_ts.shape[0]

    # global signal regression (per subject)
    global_signal_regression(fmri_task_ts)
    global_signal_regression(fmri_rest_ts)

    print(f'fMRI subjects selected: {n_fmri_subj}')
    print(f'fMRI task shape: {fmri_task_ts.shape}')
    print(f'fMRI rest shape: {fmri_rest_ts.shape}')

    return fmri_task_ts, fmri_rest_ts, fmri_rest_nsteps, selected
