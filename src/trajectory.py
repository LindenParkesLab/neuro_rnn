"""Training-trajectory analyses: how structure, dynamics and behaviour co-evolve.

Backs the ``biornn_analysis_trajectory`` and ``biornn_analysis_topology``
notebooks (paper Figs. 3, 4 and Figs. S5, S8, S9).

The heavy computation is done once by ``scripts/biornn_results_dynamics_trajectory.py``,
which walks every run of every model across training checkpoints and writes a
pickle. This module reads that pickle and reshapes it into the per-epoch arrays
the figures need; nothing here re-evaluates a network.

A note on the weight--kernel cosine
-----------------------------------
The trajectory engine stores an **uncentred** cosine between the recurrent
weights and the spatial kernel, which is the quantity the hysteresis figures use
and the one the Methods describe. :func:`src.dynamics.weight_kernel_similarity`
defaults to the *centred* form instead, which is appropriate where the question
is signed alignment against an individual's geometry. They are different
measures; read the stored value rather than recomputing, or the phase onsets
derived from it will shift.
"""

import os
import pickle

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

import src.null_utils as nu
from src.config import get_paths

# Width (in checkpoints) of the boxcar used to smooth trajectories before
# locating turning points. Onsets are read off the smoothed curve; the raw
# values are always what gets plotted.
SMOOTH_WINDOW = 5

# A run is taken to have started learning once accuracy has climbed this
# fraction of the way from the phase-III onset to its final value.
ACCURACY_ONSET_FRACTION = 0.05


def load_trajectory(model_params_name, model_dir=None):
    """Load the pickle written by the trajectory engine.

    Returns the raw dict: ``model_params``, ``args``, ``fmri_subjects``,
    ``fmri_intersubj_baselines`` and ``results`` (one entry per model row).
    """
    model_dir = model_dir or get_paths(model_params_name).model_dir
    path = os.path.join(model_dir, f'training_trajectory_{model_params_name}.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Trajectory results not found:\n  {path}\n'
            f'Run scripts/biornn_results_dynamics_trajectory.py '
            f'--model_params {model_params_name} to produce it.')
    with open(path, 'rb') as f:
        return pickle.load(f)


def class_results(trajectory, class_label_fn=None):
    """Return the per-model results, skipping rows the engine did not process."""
    return [r for r in trajectory['results'] if r is not None]


def epoch_arrays(result, fields=('accuracy',), modality='task',
                 wk_metric='cosine', subset=None):
    """Reshape one model's nested per-run/per-epoch records into arrays.

    Parameters
    ----------
    result : dict
        One entry of ``trajectory['results']``.
    fields : sequence of str
        Which scalar per-epoch fields to extract, e.g. ``'accuracy'``.
    modality : {'task', 'rest'}
        Which fMRI modality's variance-explained array to summarise.
    wk_metric : str
        Key of the stored ``weight_kernel_similarity`` dict, e.g. ``'cosine'``
        (Fig. 3a) or ``'spearman'`` (Fig. S5). Absent for models trained without
        a kernel, in which case the entry is all-NaN.
    subset : str, optional
        Node subset name (e.g. ``'bystander'``). Defaults to the whole network.

    Returns
    -------
    dict
        ``epochs`` (n_epochs,), ``ve`` (n_runs, n_epochs) subject-averaged fMRI
        variance explained, ``wk`` (n_runs, n_epochs) weight--kernel similarity,
        and one (n_runs, n_epochs) array per requested field.
    """
    epochs = np.asarray(result['sampled_epochs'], int)
    runs = result['runs']
    shape = (len(runs), epochs.size)

    out = {'epochs': epochs,
           've': np.full(shape, np.nan),
           'wk': np.full(shape, np.nan)}
    for field in fields:
        out[field] = np.full(shape, np.nan)

    ve_key = f'fmri_{modality}_ve'
    for ri, run in enumerate(runs):
        for ei, epoch in enumerate(epochs):
            record = run['epochs'].get(int(epoch))
            if record is None:
                continue

            ve = (record['node_subsets'][subset][ve_key] if subset
                  else record.get(ve_key))
            if ve is not None:
                # (n_pc, n_subjects): sum the components, then average subjects.
                out['ve'][ri, ei] = np.asarray(ve).sum(axis=0).mean()

            wk = record.get('weight_kernel_similarity')
            if wk is not None and wk_metric in wk:
                out['wk'][ri, ei] = wk[wk_metric]

            for field in fields:
                if field in record:
                    out[field][ri, ei] = record[field]
    return out


def variance_explained_z(arrays, null, modality='task'):
    """Convert subject-averaged variance explained to z against a random-subspace null.

    Applied per run and per checkpoint, using the same null and the same
    standardization as every other figure, so trajectories are directly
    comparable to the values reported elsewhere.
    """
    return nu.ve_to_z(arrays['ve'], null, modality)


def smooth(y, window=SMOOTH_WINDOW):
    """Boxcar smoothing that holds the edges rather than shrinking toward zero."""
    return uniform_filter1d(np.asarray(y, float), size=window, mode='nearest')


def phase_onsets(wk_mean, accuracy_mean, window=SMOOTH_WINDOW,
                 accuracy_fraction=ACCURACY_ONSET_FRACTION):
    """Locate the boundaries of the third training phase.

    The trajectory passes through three stages: structural similarity to the
    embedding rises while fMRI prediction is still at chance; prediction then
    rises; finally similarity falls back to an intermediate level while
    prediction stays high and the task is learned.

    Phase III begins where weight--kernel similarity peaks and starts to
    reverse. Behaviour is taken to follow once accuracy has risen
    ``accuracy_fraction`` of the way from that point to its final value.

    Parameters
    ----------
    wk_mean, accuracy_mean : (n_epochs,) arrays
        Run-averaged weight--kernel similarity and accuracy.

    Returns
    -------
    dict
        ``phase_iii`` (index of peak similarity), ``accuracy_onset``,
        ``mid_phase_iii``. Indices into the checkpoint axis.
    """
    wk_smooth = smooth(wk_mean, window)
    onset = int(np.nanargmax(wk_smooth))

    accuracy_mean = np.asarray(accuracy_mean, float)
    final = accuracy_mean[-1]
    threshold = accuracy_mean[onset] + accuracy_fraction * (final - accuracy_mean[onset])
    after = np.flatnonzero(accuracy_mean[onset:] >= threshold)
    accuracy_onset = int(onset + after[0]) if after.size else onset

    return {'phase_iii': onset,
            'accuracy_onset': accuracy_onset,
            'mid_phase_iii': (accuracy_onset + accuracy_mean.size - 1) // 2}


def turning_point(y, window=SMOOTH_WINDOW, snr=1.0, ci=None):
    """Index of the last prominent extremum of a smoothed trajectory.

    Used to start the window over which a metric is related to task accuracy,
    so that the large fluctuations early in training do not dominate.
    ``ci`` (per-point half-widths) sets a noise floor for prominence.
    """
    y_smooth = smooth(y, window)
    span = np.nanmax(y_smooth) - np.nanmin(y_smooth)
    floor = snr * np.nanmedian(ci) if ci is not None else 0.0
    prominence = max(0.15 * span, floor)

    peaks, _ = find_peaks(y_smooth, prominence=prominence)
    troughs, _ = find_peaks(-y_smooth, prominence=prominence)
    extrema = np.concatenate([peaks, troughs])
    return int(extrema.max()) if extrema.size else 0


def mean_ci(values, z=1.96, axis=0):
    """Mean and half-width of a normal-approximation confidence interval."""
    values = np.asarray(values, float)
    n = np.sum(~np.isnan(values), axis=axis)
    mean = np.nanmean(values, axis=axis)
    sem = np.nanstd(values, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))
    return mean, z * sem
