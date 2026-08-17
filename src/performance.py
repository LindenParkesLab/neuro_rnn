"""Task-performance analyses for the DMS-D training runs.

Backs the ``biornn_analysis_performance`` notebook (paper Fig. 1e, 1f and
Figs. S1, S2): loading per-class training metrics, fitting logistic learning
curves, deriving convergence epochs, and the paired statistics comparing RNN
classes.

The three RNN classes are distinguished only by two configuration choices, so
their paper-facing names are derived from those rather than hard-coded by row:

============  =============  ============  ==============
Class         mask_weights   kernel_type   Paper name
============  =============  ============  ==============
Vanilla       False          None          Vanilla RNN
Masked        True           None          Masked RNN
bioRNN        True           euclidean     bioRNN
============  =============  ============  ==============

Runs are matched by index across classes: a given run index reproduces the same
initialization, trial stream and recurrent-noise draws in every class, which is
what makes the class comparisons legitimately paired.
"""

import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import rankdata, wilcoxon

import src.utils as utils
from src.config import get_paths
from src.neural_network import ModelDataManager

from itertools import combinations

# Fraction of the fitted asymptote defining convergence, and the percentile of
# per-run convergence epochs defining a class-level training criterion.
CONV_FRAC = 0.95
CONV_PCTILE = 95

# Rounding applied when turning the slowest class criterion into the single
# epoch at which all downstream analyses are performed.
ANALYSIS_EPOCH_MARGIN = 0.05
ANALYSIS_EPOCH_ROUND = 5000

# Chance level for the DMS-D task: one of three test stimuli is the match.
CHANCE_ACCURACY = 100.0 / 3.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def class_label(mask_weights, kernel_type):
    """Return the paper-facing class name for a model configuration."""
    masked = bool(mask_weights)
    embedded = str(kernel_type) not in ('None', 'none', 'nan', '')
    if embedded:
        return 'bioRNN' if masked else 'Embedded (unmasked)'
    return 'Masked RNN' if masked else 'Vanilla RNN'


def load_performance(model_params_name, rows=(0, 1, 2),
                     model_dir=None, data_dir=None):
    """Load per-class training metrics for the requested params-CSV rows.

    The full training trace is returned and used throughout: reported
    accuracies come from the final logged epoch, and the logistic fits in
    :func:`fit_runs` use the entire trajectory. 

    Parameters
    ----------
    model_params_name : str
        Params CSV name without extension, e.g. ``'model_params_202606d'``.
    rows : sequence of int
        0-indexed rows of the CSV; the default selects the three RNN classes.
    model_dir, data_dir : str, optional
        Override the configured locations (see :mod:`src.config`).

    Returns
    -------
    list of dict
        One entry per row, each with:

        ``label``           paper-facing class name
        ``n_runs``          number of trained networks
        ``log_freq``        epochs between logged accuracy points
        ``epochs``          (n_points,) epoch index for ``accuracy``
        ``accuracy``        (n_runs, n_points) test accuracy in percent
        ``final_accuracy``  (n_runs,) accuracy at the last logged epoch --
                            the quantity reported in the paper
        ``loss_task``       (n_runs, n_epochs) per-epoch task loss
        ``loss_spatial``    (n_runs, n_epochs) per-epoch regularization loss
    """
    paths = get_paths(model_params_name)
    model_dir = model_dir or paths.model_dir
    data_dir = data_dir or paths.data_dir

    params_file = os.path.join(data_dir, model_params_name + '.csv')
    model_params, *_ = utils.get_params_dataframe(params_file, rows=list(rows))

    out = []
    for i in range(len(model_params)):
        row = model_params.iloc[i]
        outputs_file = os.path.join(model_dir, row.file_str_outputs)
        if not os.path.isfile(outputs_file):
            raise FileNotFoundError(
                f'Trained model outputs not found:\n  {outputs_file}\n'
                f'Train row {rows[i]} of {model_params_name} first, or point '
                f'model_dir in paths.yaml at the directory holding them.'
            )

        manager = ModelDataManager(outputs_file)
        n_runs, logged_epochs, _ = manager.get_info()
        final_epoch = logged_epochs[-1]

        # log_freq is recorded in the saved config; fall back to inferring it
        # from the trace length if an older run did not store it.
        log_freq = _read_log_freq(model_dir, row.file_str_outputs)

        accuracy = manager.load_key_across_runs('test_accuracy', epoch=final_epoch)
        loss_task = manager.load_key_across_runs('training_loss_task', epoch=final_epoch)
        loss_spatial = manager.load_key_across_runs('training_loss_spatial', epoch=final_epoch)

        # Drop the epoch-0 point (untrained network) and convert to percent.
        acc = np.vstack([np.asarray(a)[1:] for a in accuracy]) * 100.0
        if log_freq is None:
            log_freq = int(round((final_epoch + 1) / acc.shape[1]))
        epochs = (np.arange(acc.shape[1]) + 1) * log_freq

        out.append({
            'label': class_label(row.mask_weights, row.kernel_type),
            'n_runs': int(n_runs),
            'log_freq': int(log_freq),
            'epochs': epochs,
            'accuracy': acc,
            'final_accuracy': acc[:, -1],
            'loss_task': _stack_ragged(loss_task),
            'loss_spatial': _stack_ragged(loss_spatial),
        })
    return out


def _read_log_freq(model_dir, file_str_outputs):
    """Return ``log_freq`` from the saved config, or None if unavailable."""
    config_file = os.path.join(
        model_dir, file_str_outputs.replace('_outputs.h5', '_config.npy'))
    if not os.path.isfile(config_file):
        return None
    config = np.load(config_file, allow_pickle=True).item()
    return config.get('log_freq')


def _stack_ragged(arrays):
    """Stack per-run 1-D arrays, truncating to the shortest (runs may differ)."""
    if arrays is None or len(arrays) == 0:
        return None
    arrays = [np.asarray(a) for a in arrays]
    n = min(a.shape[0] for a in arrays)
    return np.vstack([a[:n] for a in arrays])


# ---------------------------------------------------------------------------
# Logistic learning curves (Fig. S1)
# ---------------------------------------------------------------------------

def logistic(t, a, k, t0):
    """Logistic learning curve: asymptote ``a``, rate ``k``, midpoint ``t0``."""
    return a / (1.0 + np.exp(-k * (t - t0)))


def r_squared(y, y_hat):
    """Coefficient of determination; NaN when ``y`` has zero variance."""
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - np.sum((y - y_hat) ** 2) / ss_tot


def fit_logistic(x, y, chance=0.0):
    """Fit :func:`logistic` to one accuracy trace.

    Returns a dict with ``a``, ``k``, ``t0``, ``r2``, or ``None`` if the fit
    fails or is degenerate (asymptote at/below chance, or a midpoint beyond the
    observed training window -- i.e. the run never demonstrably learned).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    a0 = max(np.nanmax(y), 1.0)
    k0 = 4.0 / (x[-1] - x[0] + 1e-9)
    above = np.flatnonzero(y >= a0 / 2.0)
    t0_0 = x[above[0]] if above.size else x[len(x) // 2]

    try:
        popt, _ = curve_fit(
            logistic, x, y, p0=[a0, k0, t0_0],
            bounds=([0.0, 0.0, x[0]], [105.0, np.inf, x[-1] * 1.5]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None

    a, k, t0 = popt
    if a <= chance or t0 > x[-1]:
        return None
    return {'a': a, 'k': k, 't0': t0, 'r2': r_squared(y, logistic(x, *popt))}


def convergence_epoch(k, t0, frac=CONV_FRAC):
    """Epoch at which a fitted curve reaches ``frac`` of its asymptote."""
    if k <= 0:
        return np.nan
    return t0 + np.log(frac / (1.0 - frac)) / k


def fit_runs(epochs, accuracy, frac=CONV_FRAC, chance=0.0,
             exclude_non_learners=True):
    """Fit every run of one class and derive its convergence epoch.

    Runs that never learned the task are excluded first. A network whose
    accuracy never rises above chance has no learning curve to characterise, so
    including it would distort the class summary and can make the logistic fit
    degenerate. 

    Parameters
    ----------
    epochs : (n_points,) array
        Epoch index of each logged accuracy value.
    accuracy : (n_runs, n_points) array
        Test accuracy in percent.
    frac : float
        Fraction of the fitted asymptote defining convergence.
    chance : float
        A fitted asymptote at or below this counts as a failed fit.
    exclude_non_learners : bool
        Drop runs whose final accuracy is at or below chance level.

    Returns
    -------
    dict
        ``run_index``    (n_kept,) index of each successfully fitted run
        ``t_conv``       (n_kept,) convergence epoch per kept run
        ``r2``           (n_kept,) fit quality per kept run
        ``params``       (n_kept, 3) fitted (a, k, t0)
        ``n_excluded``   runs dropped for never learning
        ``excluded``     their run indices
        ``n_failed_fit`` runs whose logistic fit failed or was degenerate
    """
    epochs = np.asarray(epochs, float)

    learned = (accuracy[:, -1] > CHANCE_ACCURACY if exclude_non_learners
               else np.ones(accuracy.shape[0], bool))
    excluded = np.flatnonzero(~learned)

    keep_idx, t_conv, r2, params = [], [], [], []
    for run in np.flatnonzero(learned):
        fit = fit_logistic(epochs, accuracy[run], chance=chance)
        if fit is None:
            continue
        t = convergence_epoch(fit['k'], fit['t0'], frac)
        if not np.isfinite(t):
            continue
        keep_idx.append(int(run))
        t_conv.append(t)
        r2.append(fit['r2'])
        params.append([fit['a'], fit['k'], fit['t0']])

    return {
        'run_index': np.asarray(keep_idx, int),
        't_conv': np.asarray(t_conv, float),
        'r2': np.asarray(r2, float),
        'params': np.asarray(params, float).reshape(-1, 3),
        'n_excluded': int(excluded.size),
        'excluded': excluded,
        'n_failed_fit': int(learned.sum() - len(keep_idx)),
    }


def criterion_epoch(t_conv, pctile=CONV_PCTILE):
    """Class-level training criterion: a percentile of per-run convergence."""
    t_conv = np.asarray(t_conv, float)
    t_conv = t_conv[np.isfinite(t_conv)]
    return np.percentile(t_conv, pctile) if t_conv.size else np.nan


def derive_analysis_epoch(criteria, margin=ANALYSIS_EPOCH_MARGIN,
                          round_to=ANALYSIS_EPOCH_ROUND):
    """Return the single epoch at which all downstream analyses are performed.

    Every RNN class must have reached stable performance by this point, so it
    is taken from the *slowest* class criterion, padded by ``margin`` and
    rounded up to the nearest ``round_to``.
    """
    slowest = np.nanmax(np.asarray(criteria, float))
    if not np.isfinite(slowest):
        return np.nan
    return int(np.ceil(slowest * (1.0 + margin) / round_to) * round_to)


# ---------------------------------------------------------------------------
# Paired statistics (Fig. 1f)
# ---------------------------------------------------------------------------

def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation.

    Zero differences are dropped, as in the Wilcoxon signed-rank test itself.
    Returns a value in [-1, 1]; NaN if all differences are zero.
    """
    d = np.asarray(diffs, float)
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()


def holm_bonferroni(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = p.size
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def compare_classes(fits, labels):
    """Pairwise paired comparisons of convergence epochs between classes.

    Runs are matched on ``run_index``, so only runs fitted successfully in both
    classes contribute -- preserving the paired design.

    Parameters
    ----------
    fits : sequence of dict
        Per-class outputs of :func:`fit_runs`.
    labels : sequence of str
        Class names, aligned with ``fits``.

    Returns
    -------
    list of dict
        One per pair, with ``a``, ``b``, ``n_pairs``, ``median_a``,
        ``median_b``, ``p``, ``p_holm`` and ``rank_biserial``.
    """

    results = []
    for i, j in combinations(range(len(fits)), 2):
        common = np.intersect1d(fits[i]['run_index'], fits[j]['run_index'])
        a = fits[i]['t_conv'][np.isin(fits[i]['run_index'], common)]
        b = fits[j]['t_conv'][np.isin(fits[j]['run_index'], common)]
        diffs = a - b
        p = 1.0 if np.all(diffs == 0) else wilcoxon(a, b).pvalue
        results.append({
            'a': labels[i], 'b': labels[j],
            'n_pairs': int(common.size),
            'median_a': float(np.median(a)) if a.size else np.nan,
            'median_b': float(np.median(b)) if b.size else np.nan,
            'p': float(p),
            'rank_biserial': float(rank_biserial(diffs)),
        })

    for res, p_adj in zip(results, holm_bonferroni([r['p'] for r in results])):
        res['p_holm'] = float(p_adj)
    return results


def significance_stars(p):
    """Conventional significance markers for annotating plots."""
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'
