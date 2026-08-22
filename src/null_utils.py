"""Spin-null integration helpers: resolve a null model row into its per-permutation
files, compute the fMRI variance-explained (VE) metric for a model run, and compare a
real model against a permutation null distribution.

These are deliberately notebook-agnostic so the same logic drives the dynamics
notebook and any standalone script. A null permutation is just a model trained with a
spin-permuted kernel, written to a '-null{p}' file (see train_rnn.py); here we treat
each permutation as an ordinary model and summarize them into a null distribution.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from joblib import Parallel, delayed
from scipy import stats as sstats
from sklearn.decomposition import PCA

import src.pca_utils as pca_utils
from src.neural_network import (ModelStateManager, create_rnn_and_env_for_model,
                                run_testing, run_testing_rest)


# ---------------------------------------------------------------------------
# Resolving a null row into its per-permutation model rows
# ---------------------------------------------------------------------------

_MODELS_SUFFIX = '_models.h5'
_OUTPUTS_SUFFIX = '_outputs.h5'


def _null_perms(row):
    """Number of spin permutations a model row expands to (0 if not a null row)."""
    return int(row.get('null_perms', 0) or 0)


def _base_file_str(row):
    """Base file_str (no suffix), from a 'file_str' key or by stripping 'file_str_models'."""
    fs = row.get('file_str')
    if fs is not None:
        return fs
    fsm = row['file_str_models']
    return fsm[:-len(_MODELS_SUFFIX)] if fsm.endswith(_MODELS_SUFFIX) else fsm


def null_file_strs(row):
    """List of base file_str identifiers a model row expands to.

    Normal model -> [base]; null model -> ['{base}-null0', ..., '{base}-null{N-1}'].
    Relies on the training convention that permutation p is written with the
    '-null{p}' suffix appended to the base file_str.
    """
    base = _base_file_str(row)
    n = _null_perms(row)
    return [f'{base}-null{p}' for p in range(n)] if n > 0 else [base]


def make_null_rows(null_row):
    """Expand a null model row into one row per permutation, ready for loading.

    Each returned row is a copy of ``null_row`` with file_str / file_str_models /
    file_str_outputs repointed to the permutation's '-null{p}' files and a
    ``null_index`` set. Works with the same loaders as ordinary model rows.
    """
    rows = []
    for p, fs in enumerate(null_file_strs(null_row)):
        r = null_row.copy()
        r['file_str'] = fs
        r['file_str_models'] = fs + _MODELS_SUFFIX
        r['file_str_outputs'] = fs + _OUTPUTS_SUFFIX
        r['null_index'] = p
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# fMRI variance-explained metric for a single model run
# ---------------------------------------------------------------------------

def _fit_pca(hidden_activity, n_pc):
    """Fit PCA on RNN hidden activity. Thin wrapper over :mod:`src.pca_utils`."""
    return pca_utils.fit_pca(hidden_activity, n_components=n_pc)


def _subspace_ve(fmri, subspace):
    """Fraction of one subject's fMRI variance in ``subspace``. See pca_utils."""
    return pca_utils.subspace_ve(fmri, subspace)


def _subspace_ve_all(fmri_ts, subspace):
    """Subject-averaged :func:`_subspace_ve`. See pca_utils."""
    return pca_utils.subspace_ve_all(fmri_ts, subspace)


def model_fmri_ve(rnn, dataset, fmri_task_ts, fmri_rest_ts, n_pc,
                  n_trials=100, rest_nsteps=None, baseline=1.0, seed=None):
    """fMRI VE (task & rest) for one trained run, matching the dynamics pipeline.

    Tests the RNN (task + resting-state), fits PCA on each regime's hidden
    activity, and reports the subject-averaged variance of the matched fMRI
    modality captured by that PC subspace, normalized by ``baseline``.

    ``seed`` (int) makes the evaluation deterministic -- a fixed task test battery
    and resting-state noise input -- which is required for noise-free null testing
    (e.g. the fixed-initialization spin null). ``None`` gives a stochastic
    evaluation. Returns {'task', 'rest', 'accuracy'}.
    """
    torch.set_num_threads(1)

    if rest_nsteps is None:
        rest_nsteps = fmri_rest_ts.shape[0]

    accuracy, _, _, h_task, _, _ = run_testing(
        dataset=dataset, model=rnn, n_trials=n_trials, verbose=False, test_seed=seed)
    _, h_rest, _ = run_testing_rest(
        rnn, smooth_noise=0, noise_mean=0.5, noise_sd=0.3,
        n_steps=rest_nsteps, fix_input_channels=[], seed=seed)

    pca_task = _fit_pca(h_task, n_pc)
    pca_rest = _fit_pca(h_rest, n_pc)

    ve_task = _subspace_ve_all(fmri_task_ts, pca_task)
    ve_rest = _subspace_ve_all(fmri_rest_ts, pca_rest)
    return {'task': ve_task / baseline, 'rest': ve_rest / baseline, 'accuracy': accuracy}


def runs_fmri_ve(run_models, fmri_task_ts, fmri_rest_ts, n_pc,
                 n_trials=100, rest_nsteps=None, baseline=1.0, n_jobs=-1, seed=None):
    """Per-run fMRI VE for a list of already-loaded models (e.g. the real model).

    ``run_models`` is a list of {'rnn', 'dataset'} dicts (as in trained_models_all).
    Returns {'task', 'rest', 'accuracy'} arrays, one entry per run, computed with the
    same ``model_fmri_ve`` used for the null so the comparison is apples-to-apples.
    ``seed`` makes evaluation deterministic (see ``model_fmri_ve``).
    """
    res = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(model_fmri_ve)(m['rnn'], m['dataset'], fmri_task_ts, fmri_rest_ts,
                               n_pc, n_trials, rest_nsteps, baseline, seed)
        for m in run_models)
    return {'task': np.array([r['task'] for r in res]),
            'rest': np.array([r['rest'] for r in res]),
            'accuracy': np.array([r['accuracy'] for r in res])}


# ---------------------------------------------------------------------------
# Null distribution across permutations
# ---------------------------------------------------------------------------

def _nearest_epoch(logged_epochs, epoch):
    """Resolve a target epoch to the nearest logged checkpoint (-1/None -> last)."""
    if epoch in (-1, None):
        return logged_epochs[-1]
    if epoch in logged_epochs:
        return epoch
    return logged_epochs[int(np.argmin(np.abs(np.asarray(logged_epochs) - epoch)))]


def _permutation_ve(perm_row, modeldir, device, epoch, fmri_task_ts, fmri_rest_ts,
                    n_pc, n_trials, rest_nsteps, baseline, n_runs, seed):
    """Per-permutation mean fMRI VE over its runs (one thread owns this h5 file).

    Iterates the run indices *actually stored* in the file -- which may be a
    non-contiguous set (e.g. the spanning runs of a mixed-effects null), not
    0..n_runs-1. The target epoch is resolved per run, and runs that have not
    reached it yet are skipped, so a partially-trained ensemble can be peeked at
    without crashing. Returns NaN means if no run is usable yet.
    """
    mm = ModelStateManager(os.path.join(modeldir, perm_row['file_str_models']))
    run_ids = mm.get_run_ids()
    if n_runs is not None:
        run_ids = run_ids[:n_runs]
    task, rest, use_epoch = [], [], None
    for run in run_ids:
        run_epochs = mm.get_run_epochs(run)
        if not run_epochs:
            continue
        use_epoch = _nearest_epoch(run_epochs, epoch)
        ds, rnn = create_rnn_and_env_for_model(perm_row, run, use_epoch, modeldir, device)
        ve = model_fmri_ve(rnn, ds, fmri_task_ts, fmri_rest_ts, n_pc,
                           n_trials=n_trials, rest_nsteps=rest_nsteps, baseline=baseline, seed=seed)
        task.append(ve['task'])
        rest.append(ve['rest'])
    if not task:
        return float('nan'), float('nan'), use_epoch, task, rest
    return float(np.mean(task)), float(np.mean(rest)), use_epoch, task, rest


def compute_null_distribution(null_row, fmri_task_ts, fmri_rest_ts, n_pc, modeldir, device,
                              epoch=-1, n_trials=100, rest_nsteps=None, baseline=1.0,
                              n_runs=None, n_jobs=-1, max_perms=None, seed=None):
    """Per-permutation mean fMRI VE for a null model row -> the null distribution.

    Parallelized over permutations with threads, so each '-null{p}' h5 file is read
    by exactly one worker (HDF5-safe) while the shared fMRI arrays are not copied.

    Returns {'task', 'rest'} arrays of length n_perms (per-permutation means),
    plus 'epochs_used' and 'n_perms'.
    """
    perm_rows = make_null_rows(null_row)
    if max_perms is not None:
        perm_rows = perm_rows[:max_perms]
    # processes, not threads: run_testing steps the RNN in a Python loop (GIL-bound),
    # so threads only contend. Each worker loads its own '-null{p}' h5 (process-safe)
    # and the read-only fMRI arrays are auto-memmapped by loky (not copied).
    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_permutation_ve)(pr, modeldir, device, epoch, fmri_task_ts, fmri_rest_ts,
                                 n_pc, n_trials, rest_nsteps, baseline, n_runs, seed)
        for pr in perm_rows)
    return {
        'task': np.array([r[0] for r in results]),         # per-permutation means
        'rest': np.array([r[1] for r in results]),
        'task_runs': np.array([r[3] for r in results]),    # (n_perm, R) per-run VE
        'rest_runs': np.array([r[4] for r in results]),
        'epochs_used': [r[2] for r in results],
        'n_perms': len(perm_rows),
    }


# ---------------------------------------------------------------------------
# Real-vs-null comparison
# ---------------------------------------------------------------------------
# Random-subspace null
# ---------------------------------------------------------------------------

def random_orthonormal_subspace(n_features, n_pc, rng):
    """One uniformly random ``n_pc``-dimensional subspace of R^n_features.

    Returned as ``(n_pc, n_features)`` rows, matching ``PCA.components_``.
    """
    q, _ = np.linalg.qr(rng.standard_normal((n_features, n_pc)))
    return q[:, :n_pc].T


def random_subspace_null(fmri_ts_by_modality, n_pc, n_draws=1000, paired=False,
                         seed=0, node_idx=None):
    """Null distribution of fMRI variance explained by random subspaces.

    ``n_draws`` is the number of null samples returned, in both modes -- it is
    never silently doubled or halved to suit the internal draw scheme.

    Two modes, because two different statistics need calibrating:

    ``paired=False`` (default)
        Draws ``n_draws`` single random subspaces. Calibrates a plain subspace
        VE -- one RNN's PC subspace predicting fMRI (Figs. 2c, 2d, 3a).
        Returns ``{'ve': (n_draws,)}`` per modality.

    ``paired=True``
        Draws ``n_draws`` *pairs* plus each pair's union, because the variance
        decomposition compares two predictor sets at once (Fig. 3b). Consumes
        ``2 * n_draws`` subspaces internally but still yields ``n_draws``
        samples of each decomposition statistic. Returns ``ve_a``, ``ve_b``,
        ``ve_union``, ``unique_a``, ``unique_b``, ``shared``, each ``(n_draws,)``.

    Note on naming: ``a`` and ``b`` are the two members of a pair, standing in
    for the task-driven and noise-driven RNN bases. They are unrelated to the
    ``'task'``/``'rest'`` fMRI modalities, which are keyed separately -- each
    modality gets its own null, and they are never pooled.

    Parameters
    ----------
    fmri_ts_by_modality : dict of str -> (time, nodes, subjects) array
    n_pc : int
        Subspace dimensionality; must match the real subspaces being calibrated.
    n_draws : int
        Number of null samples to return.
    paired : bool
        Draw pairs and decomposition statistics instead of single subspaces.
    seed : int
        Seeds the draw sequence, so a given call is exactly reproducible.
    node_idx : array-like of int, optional
        Restrict to a subset of nodes (e.g. bystander nodes), applied to both
        the fMRI and the subspace dimensionality.

    Returns
    -------
    dict
        ``{modality: {component: (n_draws,) subject-averaged VE}}`` plus a
        ``'meta'`` entry recording ``n_pc``, ``n_draws``, ``paired`` and ``seed``.
        Each value is already averaged over subjects, so one element is one
        null sample of the quantity being z-scored.
    """
    rng = np.random.default_rng(seed)

    modalities = {}
    for name, ts in fmri_ts_by_modality.items():
        ts = np.asarray(ts)
        modalities[name] = ts[:, node_idx, :] if node_idx is not None else ts
    n_features = next(iter(modalities.values())).shape[1]

    if paired:
        components = ('ve_a', 've_b', 've_union', 'unique_a', 'unique_b', 'shared')
    else:
        components = ('ve',)
    out = {mod: {c: np.zeros(n_draws) for c in components} for mod in modalities}

    for draw in range(n_draws):
        basis_a = random_orthonormal_subspace(n_features, n_pc, rng)
        if paired:
            basis_b = random_orthonormal_subspace(n_features, n_pc, rng)
            basis_union = pca_utils.orth_basis(np.vstack([basis_a, basis_b]))

        for mod, ts in modalities.items():
            ve_a = pca_utils.subspace_ve_all(ts, basis_a)
            if not paired:
                out[mod]['ve'][draw] = ve_a
                continue
            ve_b = pca_utils.subspace_ve_all(ts, basis_b)
            ve_u = pca_utils.subspace_ve_all(ts, basis_union)
            out[mod]['ve_a'][draw] = ve_a
            out[mod]['ve_b'][draw] = ve_b
            out[mod]['ve_union'][draw] = ve_u
            out[mod]['unique_a'][draw] = ve_u - ve_b
            out[mod]['unique_b'][draw] = ve_u - ve_a
            out[mod]['shared'][draw] = ve_a + ve_b - ve_u

    out['meta'] = {'n_pc': n_pc, 'n_draws': n_draws, 'paired': paired,
                   'seed': seed, 'n_features': n_features}
    return out


def ve_to_z(value, null, modality, component=None):
    """Standardize a variance-explained value against its matching null.

    ``component`` selects which null statistic to calibrate against, and defaults
    to the only one present for an unpaired null. A decomposition component must
    be calibrated against the null distribution of that same component: a
    ``shared`` value scored against the plain-VE null is not interpretable.
    """
    draws = null[modality]
    if component is None:
        component = 've' if 've' in draws else None
        if component is None:
            raise ValueError(
                f"this null holds {sorted(draws)}; pass component= explicitly")
    samples = np.asarray(draws[component], float)
    return (np.asarray(value, float) - samples.mean()) / samples.std(ddof=1)


# ---------------------------------------------------------------------------

def null_stats(real, null_values):
    """Compare a real metric value to a permutation null distribution.

    Returns z (vs null mean/std), one-sided p = P(null >= real) with the standard
    +1 correction, the real value's percentile in the null, and null summaries.
    """
    null = np.asarray(null_values, dtype=float)
    real = float(np.mean(real)) if np.ndim(real) else float(real)
    sd = null.std(ddof=1)
    return {
        'real': real,
        'null_mean': float(null.mean()),
        'null_std': float(sd),
        'z': float((real - null.mean()) / sd) if sd > 0 else np.nan,
        'p': float((1 + np.sum(null >= real)) / (len(null) + 1)),
        'percentile': float((null < real).mean() * 100),
        'n_null': int(len(null)),
    }


# ---------------------------------------------------------------------------
# Fixed-initialization spin null (R = 1 at one representative run)
# ---------------------------------------------------------------------------

def representative_run(real_ve_task):
    """Index of the run whose task VE is closest to the cohort mean.

    This is the run the fixed-init null is built on: the spin test then asks
    "given the geometry-permuted VE distribution, how likely is the *mean* bioRNN
    VE?", so the reference should be the run that best represents that mean.
    """
    v = np.asarray(real_ve_task, dtype=float)
    return int(np.argmin(np.abs(v - np.nanmean(v))))


def _ve_at_run(row, run_index, modeldir, device, epoch, fmri_task_ts, fmri_rest_ts,
               n_pc, n_trials, rest_nsteps, baseline, seed):
    """fMRI VE for one specific run index of a model row (nearest-epoch resolved)."""
    mm = ModelStateManager(os.path.join(modeldir, row['file_str_models']))
    _, logged_epochs, _ = mm.get_info()
    use_epoch = _nearest_epoch(logged_epochs, epoch)
    ds, rnn = create_rnn_and_env_for_model(row, run_index, use_epoch, modeldir, device)
    ve = model_fmri_ve(rnn, ds, fmri_task_ts, fmri_rest_ts, n_pc,
                       n_trials=n_trials, rest_nsteps=rest_nsteps, baseline=baseline, seed=seed)
    return ve['task'], ve['rest'], use_epoch


def compute_fixed_init_distribution(null_row, fmri_task_ts, fmri_rest_ts, n_pc, modeldir, device,
                                    epoch=-1, n_trials=100, rest_nsteps=None, baseline=1.0,
                                    n_jobs=-1, max_perms=None, seed=None):
    """Fixed-initialization spin null: one run per permutation, all trained from the
    SAME representative init (run index = ``null_row['null_run']``). Init, recurrent
    noise and training curriculum are therefore identical across permutations, so the
    only thing that varies is the embedding geometry (sigma_train = 0 by design).

    Each '-null{p}' file is read by one thread (HDF5-safe). Returns {'task', 'rest'}
    VE arrays (one entry per permutation), plus 'run_index', 'epochs_used', 'n_perms'.
    """
    run_index = int(null_row.get('null_run', 0) or 0)
    perm_rows = make_null_rows(null_row)
    if max_perms is not None:
        perm_rows = perm_rows[:max_perms]
    # processes, not threads (see compute_null_distribution): GIL-bound RNN stepping
    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        delayed(_ve_at_run)(pr, run_index, modeldir, device, epoch,
                            fmri_task_ts, fmri_rest_ts, n_pc, n_trials,
                            rest_nsteps, baseline, seed)
        for pr in perm_rows)
    return {
        'task': np.array([r[0] for r in results]),
        'rest': np.array([r[1] for r in results]),
        'run_index': run_index,
        'epochs_used': [r[2] for r in results],
        'n_perms': len(perm_rows),
    }


def fixed_init_spin_test(real_row, null_row, fmri_task_ts, fmri_rest_ts, n_pc, modeldir, device,
                         epoch=-1, n_trials=100, rest_nsteps=None, baseline=1.0,
                         n_jobs=-1, max_perms=None, seed=None):
    """End-to-end fixed-init spin test for task and rest fMRI VE.

    The reference is the real model's run ``null_row['null_run']`` (the representative
    run, e.g. from :func:`representative_run`); the null is the fixed-init ensemble
    scored at that same run index. ``real_row`` is the ordinary (unpermuted) euclidean
    model row; ``null_row`` is its null counterpart. Returns {'task', 'rest'} ->
    ``null_stats`` dicts (each with the raw 'null_values' attached), plus 'run_index'
    and 'real_epoch'.
    """
    run_index = int(null_row.get('null_run', 0) or 0)
    real_task, real_rest, real_epoch = _ve_at_run(
        real_row, run_index, modeldir, device, epoch, fmri_task_ts, fmri_rest_ts,
        n_pc, n_trials, rest_nsteps, baseline, seed)
    null = compute_fixed_init_distribution(
        null_row, fmri_task_ts, fmri_rest_ts, n_pc, modeldir, device, epoch=epoch,
        n_trials=n_trials, rest_nsteps=rest_nsteps, baseline=baseline,
        n_jobs=n_jobs, max_perms=max_perms, seed=seed)
    return {
        'run_index': run_index,
        'real_epoch': real_epoch,
        'task': {**null_stats(real_task, null['task']), 'null_values': null['task']},
        'rest': {**null_stats(real_rest, null['rest']), 'null_values': null['rest']},
    }


def mixed_null_test(null_runs, real_runs, n_boot=2000, seed=0, use_mixedlm=True,
                    crossed=False):
    """Variance-components test of a real model against a spin null, separating
    training noise from genuine spatial spread.

    ``crossed=True`` fits ``VE ~ run + (1|perm)`` -- the run (column) main effect
    is removed before estimating components, so sigma_train is the pure run x
    geometry residual and sigma_geom is not deflated by a deliberate run spread
    (essential when the R runs were chosen to span the VE distribution, else the
    spanning spread masquerades as training noise). ``crossed=False`` keeps the
    balanced one-way model ``VE ~ 1 + (1|perm)``.

    ``null_runs`` is the (n_perm, R) matrix of per-run VE for the permuted
    geometries; ``real_runs`` is the 1-D array of per-run VE for the real model.

    The model is VE_pr = mu + b_p + eps_pr, with b_p ~ N(0, sigma_geom^2) the
    geometry effect and eps ~ N(0, sigma_train^2) training noise. For a balanced
    design this closed form IS the REML solution of ``VE ~ 1 + (1|permutation)``.

    Headline test (#1, outlier): is the real geometry beyond the *distribution* of
    permuted geometries -> z = (real_mean - mu) / sigma_geom. Also reports the
    Wald contrast (#2): real mean vs the permuted-population mean. A cluster
    bootstrap (resampling permutations and real runs) gives a CI on the headline z.
    """
    null_runs = np.asarray(null_runs, dtype=float)
    real_runs = np.asarray(real_runs, dtype=float)
    N, R = null_runs.shape
    perm_means = null_runs.mean(axis=1)
    mu = float(perm_means.mean())

    def _components(nr):
        pm = nr.mean(axis=1)
        Nn, Rr = nr.shape
        if crossed:                                          # VE ~ run + (1|perm)
            gm = nr.mean(); rm = nr.mean(axis=0)             # grand mean, run (col) means
            ss_perm = Rr * np.sum((pm - gm) ** 2)
            ss_resid = np.sum((nr - pm[:, None] - rm[None, :] + gm) ** 2)
            ms_perm = ss_perm / (Nn - 1)
            ms_resid = ss_resid / ((Nn - 1) * (Rr - 1))     # run main effect removed
            return pm.mean(), float(ms_resid), max((ms_perm - ms_resid) / Rr, 0.0)
        msw = float(np.mean(nr.var(axis=1, ddof=1)))        # within = training-noise var
        msb = float(Rr * pm.var(ddof=1))                    # between (R reps)
        return pm.mean(), msw, max((msb - msw) / Rr, 0.0)   # mu, s2_train, s2_geom

    _, s2_train, s2_geom = _components(null_runs)
    st, sg = np.sqrt(s2_train), np.sqrt(s2_geom)
    icc = s2_geom / (s2_geom + s2_train) if (s2_geom + s2_train) > 0 else np.nan

    real_mean = float(real_runs.mean()); n_real = len(real_runs)
    beta = real_mean - mu

    # #1 outlier (headline): real vs the geometry distribution
    z1 = beta / sg if sg > 0 else np.inf
    p1 = float(sstats.norm.sf(z1))

    # #2 Wald contrast: real mean vs permuted-population mean
    se_mu = np.sqrt(s2_geom / N + s2_train / (N * R))
    se_real = np.sqrt(real_runs.var(ddof=1) / n_real)
    se_beta = np.sqrt(se_mu ** 2 + se_real ** 2)
    z2 = beta / se_beta if se_beta > 0 else np.inf
    p2 = float(sstats.norm.sf(z2))

    # cluster bootstrap CI on the headline z (resample permutations and real runs).
    # Also keep the per-resample (mu, sigma_geom) so the figure can draw the
    # *uncertainty* of the geometry null rather than a falsely-crisp curve.
    rng = np.random.default_rng(seed)
    zb, boot_mu, boot_sg, n_degenerate = [], [], [], 0
    for _ in range(n_boot):
        mu_b, _, s2g_b = _components(null_runs[rng.integers(0, N, N)])
        boot_mu.append(mu_b)
        boot_sg.append(float(np.sqrt(max(s2g_b, 0.0))))
        if s2g_b <= 0:
            n_degenerate += 1
            continue
        rb = real_runs[rng.integers(0, n_real, n_real)].mean()
        zb.append((rb - mu_b) / np.sqrt(s2g_b))
    zb = np.array(zb)
    z1_ci = (float(np.percentile(zb, 2.5)), float(np.percentile(zb, 97.5))) if len(zb) else (np.nan, np.nan)
    p1_ci = (float(sstats.norm.sf(z1_ci[1])), float(sstats.norm.sf(z1_ci[0]))) if len(zb) else (np.nan, np.nan)

    resid_within = (null_runs - perm_means[:, None]).ravel()
    out = {
        'mu': mu, 'real_mean': real_mean, 'beta': beta,
        'sigma_train': float(st), 'sigma_geom': float(sg), 'icc': float(icc),
        'z_outlier': float(z1), 'p_outlier': p1,
        'z_outlier_ci': z1_ci, 'p_outlier_ci': p1_ci,
        'z_wald': float(z2), 'p_wald': p2,
        'n_perms': N, 'R': R, 'n_real': n_real,
        'boot_frac_geom_zero': n_degenerate / n_boot,
        'boot_mu': np.array(boot_mu),                # per-resample geometry-null center
        'boot_sigma_geom': np.array(boot_sg),        # per-resample geometry-null width
        'shapiro_geom_p': float(sstats.shapiro(perm_means).pvalue) if 3 <= N <= 5000 else np.nan,
        'shapiro_train_p': float(sstats.shapiro(resid_within[:5000]).pvalue) if len(resid_within) >= 3 else np.nan,
    }
    if crossed:
        run_eff = null_runs.mean(axis=0) - null_runs.mean()   # estimated run main effects
        out['run_effects'] = run_eff.tolist()
        out['sigma_run'] = float(np.sqrt(np.mean(run_eff ** 2)))
    if use_mixedlm:
        try:
            df = pd.DataFrame({'ve': null_runs.ravel(),
                               'perm': np.repeat(np.arange(N), R),
                               'run': np.tile(np.arange(R), N)})
            formula = 've ~ C(run)' if crossed else 've ~ 1'
            m = smf.mixedlm(formula, df, groups='perm').fit(reml=True, method='lbfgs')
            out['mixedlm_sigma_geom'] = float(np.sqrt(max(float(m.cov_re.iloc[0, 0]), 0.0)))
            out['mixedlm_sigma_train'] = float(np.sqrt(m.scale))
        except Exception as e:
            out['mixedlm_error'] = str(e)
    return out
