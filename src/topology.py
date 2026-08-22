import numpy as np
import bct
from scipy import stats


def threshold_adj(A, q=0.8, abs=True, fill_diag=True, binarize=True):
    if abs:
        A = np.abs(A)

    if fill_diag:
        np.fill_diagonal(A, 0)

    thresh = np.quantile(A, q=q)
    mask = A >= thresh

    A_out = A.copy()
    A_out[~mask] = 0
    if binarize:
        A_out[mask] = 1

    return A_out


def get_norm_rc(A, n_perms=1000, directed=True, weighted=True):
    if directed == True:
        _, _, degree = bct.degrees_dir(A)
        kmax = int(np.max(degree))
        if weighted == True:
            R = bct.rich_club_wd(A, klevel=kmax)
        elif weighted == False:
            R, _, _ = bct.rich_club_bd(A, klevel=kmax)

        if n_perms > 0:
            R_perm = np.zeros((n_perms, kmax))
            for i in np.arange(n_perms):
                np.random.seed(i)
                A_rand, _ = bct.randmio_dir(A, itr=5)
                if weighted == True:
                    R_perm[i, :] = bct.rich_club_wd(A_rand, klevel=kmax)
                elif weighted == False:
                    R_perm[i, :], _, _ = bct.rich_club_bd(A_rand, klevel=kmax)
    elif directed == False:
        pass  # linden, fix this

    # compute normalized rich club coefficient
    if n_perms > 0:
        R_norm = np.divide(R, np.nanmean(R_perm, axis=0))

        # compute p values
        p_val = np.zeros(kmax)
        for i in np.arange(kmax):
            p_val[i] = np.nanmean(R[i] <= R_perm[:, i])

    if n_perms > 0:
        return R, R_perm, R_norm, p_val
    else:
        return R


def _dist_stats(x, q=0.8):
    """Return (mean, skewness, top-quartile mean) of a 1D node-level vector."""
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    mean = float(np.nanmean(x)) if valid.size else np.nan
    skew = float(stats.skew(valid)) if valid.size > 2 else np.nan
    if valid.size:
        thr = np.quantile(valid, q)
        topq = float(np.nanmean(x[x >= thr]))
    else:
        topq = np.nan
    return mean, skew, topq


def compute_topology_metrics(A_raw, q=0.8, rc_perms=1000):
    """Compute segregation + centrality/hub metrics on a weighted directed graph.

    The raw (signed) weight matrix ``A_raw`` is reduced to an absolute-valued,
    zero-diagonal, weighted graph keeping the top ``1 - q`` quantile of edges
    (via :func:`threshold_adj`, matching the existing convention).  Returns a
    flat dict of scalar metrics.  Rich-club normalization uses ``rc_perms``
    permutations; set ``rc_perms=0`` (or None) to skip it.
    """
    A_raw = np.nan_to_num(np.asarray(A_raw, dtype=float),
                          nan=0.0, posinf=0.0, neginf=0.0)
    A = threshold_adj(A_raw.copy(), q=q, abs=True, fill_diag=True, binarize=False)

    out = {}

    # ---- Segregation ----
    ci, Q = bct.modularity_dir(A)
    out['modularity_q'] = float(Q)
    out['n_communities'] = int(len(np.unique(ci)))
    out['clustering_mean'] = float(np.nanmean(bct.clustering_coef_wd(A)))
    out['participation_mean'] = float(np.nanmean(bct.participation_coef(A, ci)))

    # ---- Centrality / hubs ----
    _, _, degree = bct.degrees_dir(A)
    # degree-distribution skewness (skew of histogram counts; existing convention)
    _, counts = np.unique(degree, return_counts=True)
    out['degree_skewness'] = float(stats.skew(counts)) if counts.size > 2 else np.nan
    dthr = np.quantile(degree, q)
    high_deg = degree >= dthr
    out['degree_topq_mean'] = float(np.nanmean(degree[high_deg]))

    strength = bct.strengths_dir(A)                 # total (in + out) strength
    out['strength_in_mean'] = float(np.nanmean(np.sum(A, axis=0)))
    out['strength_out_mean'] = float(np.nanmean(np.sum(A, axis=1)))
    s_mean, s_skew, _ = _dist_stats(strength, q=q)
    out['strength_mean'] = s_mean
    out['strength_skewness'] = s_skew
    # mean strength of high-degree (hub) nodes, matching existing convention
    out['strength_topq_mean'] = float(np.nanmean(strength[high_deg]))

    # weighted betweenness on the connection-length matrix (invert weights)
    bw = bct.betweenness_wei(bct.invert(A))
    b_mean, b_skew, _ = _dist_stats(bw, q=q)
    out['betweenness_mean'] = b_mean
    out['betweenness_skewness'] = b_skew

    # ---- Rich club (normalized, directed weighted) ----
    if rc_perms and rc_perms > 0:
        try:
            _, _, R_norm, _ = get_norm_rc(A, n_perms=rc_perms,
                                          directed=True, weighted=True)
            R_norm = np.asarray(R_norm, dtype=float)
            R_norm[~np.isfinite(R_norm)] = np.nan
            out['rich_club_norm_mean'] = float(np.nanmean(R_norm))
        except Exception:
            out['rich_club_norm_mean'] = np.nan
    else:
        out['rich_club_norm_mean'] = np.nan

    return out


def compute_reference_similarity(A_rnn_raw, ref_mat):
    """Matrix-level similarity between an RNN graph and a reference matrix.

    Both are reduced to absolute, symmetrized, zero-diagonal form, then compared
    over off-diagonal edges (spearman, cosine) and node-strength profiles
    (spearman).  References (spatial kernel, fMRI FC) are symmetric; the RNN is
    symmetrized so the comparison is apples-to-apples.
    """
    def _prep(M):
        M = np.abs(np.nan_to_num(np.asarray(M, dtype=float),
                                 nan=0.0, posinf=0.0, neginf=0.0))
        M = (M + M.T) / 2.0
        np.fill_diagonal(M, 0.0)
        return M

    A = _prep(A_rnn_raw)
    R = _prep(ref_mat)
    off = ~np.eye(A.shape[0], dtype=bool)
    a, r = A[off], R[off]

    denom = np.linalg.norm(a) * np.linalg.norm(r)
    return {
        'edge_spearman': float(stats.spearmanr(a, r).correlation),
        'edge_cosine': float(np.dot(a, r) / denom) if denom > 0 else np.nan,
        'strength_spearman': float(
            stats.spearmanr(np.sum(A, axis=1), np.sum(R, axis=1)).correlation),
    }


def parcel_distances(datadir, hidden_size=100, atlas_parcels=200):
    """Euclidean distances between parcel centroids, in the atlas coordinate units.

    These are the raw inter-regional distances, **not** the normalized kernel the
    networks were regularized with: connection lengths are a physical quantity
    and should be read on the atlas scale.
    """
    import os
    import pandas as pd
    from scipy.spatial import distance

    centroids = pd.read_csv(
        os.path.join(datadir, f'schaefer{atlas_parcels}_centroids.csv'))[:hidden_size]
    return distance.squareform(
        distance.pdist(centroids.set_index('ROI Name'), 'euclidean'))


def connection_length_topq(weights, distances, q=0.8):
    """Mean physical length of the strongest edges.

    Ranks the off-diagonal, non-zero connections by ``|weight|``, keeps the
    strongest ``1 - q`` of them -- the same edges the other topology metrics are
    computed on -- and returns the mean Euclidean distance those edges span.

    Rising values mean the network is investing in long-range connections, which
    a purely distance-penalized network would avoid.

    Parameters
    ----------
    weights : (n, n) array
        Recurrent weight submatrix for the nodes of interest.
    distances : (n, n) array
        Matching inter-node distances, e.g. from :func:`parcel_distances`.
    q : float
        Quantile threshold; 0.8 keeps the strongest 20%.

    Returns
    -------
    float
        NaN if no non-zero edges survive.
    """
    magnitude = np.abs(np.asarray(weights, float)).copy()
    np.fill_diagonal(magnitude, 0.0)

    off_diagonal = ~np.eye(magnitude.shape[0], dtype=bool)
    w = magnitude[off_diagonal]
    d = np.asarray(distances, float)[off_diagonal]

    nonzero = w > 0
    if not nonzero.any():
        return np.nan
    w, d = w[nonzero], d[nonzero]

    keep = w >= np.quantile(w, q)
    return float(np.mean(d[keep])) if keep.any() else np.nan
