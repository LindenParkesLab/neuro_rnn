"""Low-dimensional subspace tools shared by the dynamics analyses.

One canonical implementation of the PCA and subspace-variance machinery used to
summarise RNN hidden-state dynamics and to ask how much empirical fMRI variance
a given subspace captures. Both the trajectory script and the dynamics notebook
import from here, so the null distributions and the measurements they calibrate
are computed by exactly the same code.

Two conventions are fixed here and should not be varied per caller:

* **Centering.** Data are centered on their own temporal mean before projection
  (``centering_mean='data'``), not on the mean stored in the fitted PCA. The
  fMRI being projected is a different dataset from the RNN activity the PCA was
  fitted on, so its own mean is the meaningful origin.
* **Solver.** PCA always uses an exact, deterministic solver -- never the
  randomized one, whose results vary between calls. See :func:`exact_solver`.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


def exact_solver(n_samples, n_features):
    """Return an exact, deterministic ``svd_solver`` suited to the data shape.

    Both options are exact and reproducible; they differ only in cost.
    ``covariance_eigh`` forms the feature covariance and is far cheaper when
    samples greatly outnumber features -- the regime for concatenated
    hidden-state and fMRI time series -- while ``full`` is safe for any shape.

    Never returns ``'randomized'``, and never defers to ``'auto'``: the
    heuristic behind ``'auto'`` has changed across scikit-learn versions, and a
    version bump silently changing solver would be a reproducibility hazard.
    """
    return 'covariance_eigh' if n_samples >= 10 * n_features else 'full'


def stack_trials(data):
    """Flatten a list of ``(time, nodes)`` trials into one ``(time*trials, nodes)`` array.

    Arrays are returned unchanged, so callers can pass either form.
    """
    if isinstance(data, list):
        n_trials = len(data)
        n_timepoints, n_nodes = data[0].shape
        return np.reshape(np.asarray(data), (n_trials * n_timepoints, n_nodes))
    return np.asarray(data)


def _ve_dict(ratios):
    """Package per-component variance ratios the way the analyses consume them."""
    ratios = np.asarray(ratios, float)
    return {
        'each': ratios,
        'cumul': np.cumsum(ratios),
        'total': float(np.sum(ratios)),
    }


def fit_pca(data, n_components=5, svd_solver=None, return_variance=False):
    """Fit PCA on hidden-state or fMRI time series.

    Parameters
    ----------
    data : list of (time, nodes) arrays, or a single (samples, nodes) array
    n_components : int
        Number of components to retain (the analyses use 5).
    svd_solver : str, optional
        Overrides the solver chosen by :func:`exact_solver`.
    return_variance : bool
        Also return the explained-variance dict.

    Returns
    -------
    ``PCA``, or ``(PCA, dict)`` when ``return_variance`` is set.
    """
    X = stack_trials(data)
    solver = svd_solver or exact_solver(*X.shape)
    pca = PCA(n_components=n_components, svd_solver=solver).fit(X)
    if return_variance:
        return pca, _ve_dict(pca.explained_variance_ratio_)
    return pca


def basis_of(subspace):
    """Return a ``(k, n_nodes)`` basis from a fitted PCA or a raw array."""
    return getattr(subspace, 'components_', None) if hasattr(subspace, 'components_') \
        else np.asarray(subspace)


def pca_projection(data, subspace, centering_mean='data'):
    """Variance of ``data`` captured by each direction of ``subspace``.

    Returns the ``{'each', 'cumul', 'total'}`` dict. ``subspace`` may be a fitted
    PCA or a ``(k, n_nodes)`` basis.
    """
    X = stack_trials(data)
    components = basis_of(subspace)

    if centering_mean == 'data':
        X = X - np.mean(X, axis=0)
    elif centering_mean == 'pca':
        X = X - subspace.mean_
    else:
        raise ValueError(
            f"centering_mean must be 'data' or 'pca', got {centering_mean!r}")

    projected = X @ components.T
    total = np.sum(X ** 2)
    return _ve_dict(np.sum(projected ** 2, axis=0) / total)


def subspace_ve(data, subspace):
    """Fraction of one subject's temporal variance lying in ``subspace``.

    Equivalent to :func:`pca_projection`'s ``'total'``, without building the
    per-component breakdown.
    """
    components = basis_of(subspace)
    Xc = data - data.mean(axis=0, keepdims=True)
    projected = Xc @ components.T
    return float(np.sum(projected ** 2) / np.sum(Xc ** 2))


def subspace_ve_all(fmri_ts, subspace):
    """Subject-averaged :func:`subspace_ve` for ``(time, nodes, subjects)`` data.

    One einsum pass rather than a Python loop over subjects, so the inner work
    stays in BLAS and parallelizes across joblib threads.
    """
    components = basis_of(subspace)
    Xc = fmri_ts - fmri_ts.mean(axis=0, keepdims=True)                    # (T,N,S)
    projected = np.einsum('tns,kn->tks', Xc, components, optimize=True)   # (T,k,S)
    num = np.einsum('tks,tks->s', projected, projected)                   # (S,)
    den = np.einsum('tns,tns->s', Xc, Xc)                                 # (S,)
    return float(np.mean(num / den))


def subspace_ve_per_subject(fmri_ts, subspace):
    """Per-subject :func:`subspace_ve` for ``(time, nodes, subjects)`` data.

    The un-averaged counterpart of :func:`subspace_ve_all`, needed when the
    question is about individual subjects rather than the group -- e.g. whether
    a network that resembles *this* subject's geometry predicts *this*
    subject's activity.

    Returns
    -------
    (n_subjects,) array
    """
    components = basis_of(subspace)
    Xc = fmri_ts - fmri_ts.mean(axis=0, keepdims=True)                    # (T,N,S)
    projected = np.einsum('tns,kn->tks', Xc, components, optimize=True)   # (T,k,S)
    num = np.einsum('tks,tks->s', projected, projected)
    den = np.einsum('tns,tns->s', Xc, Xc)
    return num / den


def orth_basis(matrix, rtol=1e-10):
    """Orthonormal basis for the row space of ``matrix``, as ``(k, n_nodes)``.

    Used to pool two subspaces (e.g. task- and noise-driven PCs) into a single
    basis spanning everything either can explain.
    """
    M = np.asarray(matrix, float)
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    keep = s > (s.max() * rtol if s.size and s.max() > 0 else 0)
    return vt[keep]


def intersubj_ve_baseline(fmri_ts, n_pc, reduce='median'):
    """Variance one subject's PC subspace captures in *other* subjects' data.

    An empirical ceiling: how much of a subject's fMRI variance any equally sized
    subspace derived from real brain data can explain. Used to normalise the
    model's variance explained.

    Parameters
    ----------
    fmri_ts : (time, nodes, subjects) array
    n_pc : int
        Subspace dimensionality, matched to the model subspaces.
    reduce : {'median', 'mean'}
        How to summarise the off-diagonal subject-by-subject matrix.

    Returns
    -------
    float
    """
    n_subj = fmri_ts.shape[2]
    ve = np.full((n_subj, n_subj), np.nan)
    for i in range(n_subj):
        pca = fit_pca(fmri_ts[:, :, i], n_components=n_pc)
        for j in range(n_subj):
            if i == j:
                continue
            ve[i, j] = subspace_ve(fmri_ts[:, :, j], pca)

    off_diagonal = ve[~np.isnan(ve)]
    return float(np.median(off_diagonal) if reduce == 'median'
                 else np.mean(off_diagonal))


def align_and_average_components(pca_list, n_components=5, return_diagnostics=False):
    """Average PC loadings across runs or subjects, resolving order and sign.

    PCA determines components only up to (a) an arbitrary sign and (b) ordering
    among components of similar variance. Independently fitted datasets --
    different training runs, different subjects -- therefore need aligning
    before their loadings can be averaged, or genuinely shared structure is
    blended away.

    Each set is matched to the first one by optimal one-to-one assignment
    (Hungarian algorithm) on absolute cosine similarity, then sign-flipped to
    agree. Every set is included; match quality is reported rather than used to
    exclude, so poor matches are visible without introducing selection bias.

    Parameters
    ----------
    pca_list : sequence
        Fitted PCA objects or ``(n_pc, n_nodes)`` loading arrays.
    n_components : int
        Components to align and average.
    return_diagnostics : bool
        Also return per-set match quality.

    Returns
    -------
    (n_components, n_nodes) array, or ``(array, dict)`` with diagnostics.
    """
    reference = basis_of(pca_list[0])[:n_components]

    aligned, quality = [], []
    for item in pca_list:
        components = np.array(basis_of(item)[:n_components], dtype=float)

        # Optimal assignment of this set's components to the reference.
        cosine = np.abs(reference @ components.T)
        _, assignment = linear_sum_assignment(-cosine)

        matched = np.zeros_like(components)
        matched_quality = np.zeros(n_components)
        for k in range(n_components):
            matched[k] = components[assignment[k]]
            matched_quality[k] = cosine[k, assignment[k]]
            if reference[k] @ matched[k] < 0:
                matched[k] *= -1
        aligned.append(matched)
        quality.append(matched_quality)

    mean_components = np.mean(aligned, axis=0)
    if not return_diagnostics:
        return mean_components

    quality = np.asarray(quality)
    return mean_components, {
        'match_quality': quality,
        'mean_cosine': quality.mean(axis=0),
        'min_cosine': quality.min(axis=0),
    }
