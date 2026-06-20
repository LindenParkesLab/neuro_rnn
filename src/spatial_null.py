"""Spatial-null tools for testing geometry-beyond-autocorrelation.

The question these support: does a spatial kernel's *specific geometry* carry
fMRI-relevant structure beyond its spatial autocorrelation (SA)? The test is a
spin null (Alexander-Bloch / Vazquez-Rodriguez): rotate the parcels on the
FreeSurfer sphere and reassign each rotated parcel to a distinct original parcel
(optimal bijective assignment), giving a permutation that preserves SA while
scrambling the specific cortical layout. Applied to a kernel's gradient subspace
as U[perm], it yields an SA-matched surrogate basis.

If the real kernel's gradient subspace explains fMRI variance no better than the
spun surrogates, the alignment is "SA / smoothness" (H1). If it explains more,
the specific geometry matters (H2) and is worth the cost of training surrogate
models.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _random_rotation(rng):
    """Uniformly random proper 3D rotation (det = +1) via QR of a Gaussian."""
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(R))            # fix QR sign ambiguity -> Haar measure
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]                 # enforce a rotation, not a reflection
    return Q


def spin_permutations(sphere_coords, n_perm, seed=0):
    """SA-preserving permutations by random rotation + optimal reassignment.

    Each parcel (a point on the sphere) is rotated by a random rotation, then
    matched one-to-one to the original parcels by maximizing total cosine
    similarity (Hungarian assignment) so the result is a true bijection -- a
    valid permutation that can index a kernel as ``K[np.ix_(perm, perm)]``.

    Parameters
    ----------
    sphere_coords : (n_nodes, 3) array
        Parcel coordinates on the (FreeSurfer) sphere.
    n_perm : int
    seed : int

    Returns
    -------
    (n_perm, n_nodes) int array of permutations.
    """
    X = np.asarray(sphere_coords, float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perms = np.empty((n_perm, n), dtype=int)
    for p in range(n_perm):
        Xr = X @ _random_rotation(rng).T
        # maximize total cosine similarity -> minimize negative similarity
        _, col = linear_sum_assignment(-(Xr @ X.T))
        perms[p] = col
    return perms


def gradient_modes(kernel, k, drop_constant=True):
    """Top-k spatial gradient eigenmodes of a similarity ``kernel`` (n, n).

    The kernel is double-centered (J K J) before eigendecomposition so the
    trivial constant mode is removed -- the columns are the principal spatial
    gradients, analogous to connectome gradients. Centering commutes with node
    permutation, so a spun surrogate basis is exactly ``modes[perm]``.

    Returns (n, k) array of orthonormal gradient maps (largest eigenvalue first).
    """
    K = np.asarray(kernel, float)
    n = K.shape[0]
    if drop_constant:
        J = np.eye(n) - np.ones((n, n)) / n
        K = J @ K @ J
    w, V = np.linalg.eigh((K + K.T) / 2.0)
    order = np.argsort(w)[::-1][:k]
    return V[:, order]


def basis_fmri_ve(fmri_ts, basis):
    """Mean (over subjects) fraction of fMRI variance captured by a fixed spatial
    ``basis`` (n_nodes, k) with orthonormal columns.

    Mirrors the project-onto-fixed-subspace VE used for the RNN PCs: per subject,
    center over time, project the spatial patterns onto ``basis``, and take the
    summed projected variance as a fraction of total variance.

    Parameters
    ----------
    fmri_ts : (time, n_nodes, n_subj) array
    basis   : (n_nodes, k) array, orthonormal columns
    """
    ves = []
    for si in range(fmri_ts.shape[2]):
        X = fmri_ts[:, :, si]
        Xc = X - X.mean(axis=0, keepdims=True)
        proj = Xc @ basis
        total = np.sum(Xc ** 2)
        ves.append(np.sum(proj ** 2) / total)
    return float(np.mean(ves))


def spin_null_ve(fmri_ts, modes, perms):
    """fMRI VE for an ensemble of spun (SA-matched) versions of ``modes``.

    Returns a (n_perm,) array of VE values, one per spin permutation.
    """
    return np.array([basis_fmri_ve(fmri_ts, modes[perm]) for perm in perms])


def spin_pvalue(real_ve, null_ve):
    """One-sided p: P(null >= real), with the standard +1 correction."""
    null_ve = np.asarray(null_ve)
    return (1 + np.sum(null_ve >= real_ve)) / (len(null_ve) + 1)
