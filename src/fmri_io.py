"""Shared fMRI loading and preprocessing utilities.

The HCP task/rest fMRI pipeline and the spatial kernels built from the atlas:
seeded common-subject selection, global signal regression, kernel loading, and
the combined task/rest loader.

Every analysis loads its fMRI through here, so they all see the same subjects in
the same order -- which is what allows variance-explained values computed in
different notebooks to be compared subject for subject.

Directory locations come from :mod:`src.config` (``paths.yaml``); ``get_paths``
is re-exported here for convenience.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial import distance

import src.utils as utils
from src.config import get_paths   # noqa: F401  (re-exported for callers)


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

def kernel_distance_matrices(datadir, hidden_size=100):
    """Normalized inter-node **distance** matrices, one per spatial kernel.

    The companion to :func:`load_kernels`, which returns the ``1 - distance``
    similarity view. Both are needed: training penalizes weights that disagree
    with *proximity*, while some analyses want the raw distances. Ask for the
    one you mean rather than converting by hand, so the sign convention stays
    explicit at the call site.
    """
    centroids = pd.read_csv(
        os.path.join(datadir, 'schaefer200_centroids.csv'))[:hidden_size]
    dist = distance.squareform(
        distance.pdist(centroids.set_index('ROI Name'), 'euclidean'))

    brain_maps = {
        'sa_axis': 'schaefer200_sa-axis.npy',
        'ut_axis': 'schaefer200_ut-axis.npy',
        'sf_axis': 'schaefer200_cyto.npy',
    }
    out = {'euclidean': utils.normalize_x(dist)}
    for name, fname in brain_maps.items():
        brain_map = np.load(os.path.join(datadir, fname))[:hidden_size]
        out[name] = utils.normalize_x(utils.get_brainmap_distance(brain_map))
    return out


def load_kernels(datadir, hidden_size=100):
    """Spatial kernel **similarity** matrices (``1 - distance``).

    See :func:`kernel_distance_matrices` for the distance form.
    """
    return {name: 1 - dist
            for name, dist in kernel_distance_matrices(datadir, hidden_size).items()}


# ---------------------------------------------------------------------------
# Task + rest fMRI loader
# ---------------------------------------------------------------------------

def load_fmri_data(datadir, fmridir, n_fmri_subj, hidden_size=100,
                   apply_gsr=True, pick_random_subjects=True, subject_seed=42,
                   verbose=True):
    """Load task and resting-state fMRI for a common set of subjects.

    Parameters
    ----------
    datadir, fmridir : str
        Atlas directory and the directory holding the HCP files. ``fmridir``
        typically points outside the repository -- see ``data_private/README``.
    n_fmri_subj : int
        Number of subjects to draw from those present in both task and rest.
    hidden_size : int
        Number of parcels to keep, matching the RNN hidden layer (100 = left
        hemisphere of the Schaefer-200 atlas).
    apply_gsr : bool
        Regress the global signal out of both modalities.
    pick_random_subjects : bool
        Draw the subject set at random (seeded) rather than taking the first
        ``n_fmri_subj`` common subjects.
    subject_seed : int
        Seeds that draw. Every caller using the same seed gets the identical
        subject set, which is what lets variance-explained values computed in
        different notebooks be compared subject-for-subject.
    verbose : bool
        Print the resulting shapes.

    Returns
    -------
    (task_ts, rest_ts, rest_nsteps, subjects)
        Time series as ``(time, nodes, subjects)``.
    """
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
        fmri_task_subjnames, fmri_rest_subjnames, n_fmri_subj,
        seed=subject_seed, pick_random=pick_random_subjects)
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
    if apply_gsr:
        global_signal_regression(fmri_task_ts)
        global_signal_regression(fmri_rest_ts)

    if verbose:
        print(f'fMRI subjects selected: {n_fmri_subj} '
              f'({"seeded random" if pick_random_subjects else "first N"}, '
              f'seed={subject_seed}); GSR={"on" if apply_gsr else "off"}')
        print(f'fMRI task shape: {fmri_task_ts.shape}')
        print(f'fMRI rest shape: {fmri_rest_ts.shape}')

    return fmri_task_ts, fmri_rest_ts, fmri_rest_nsteps, selected
