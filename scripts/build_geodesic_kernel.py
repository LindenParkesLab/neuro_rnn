"""Precompute the geodesic spatial kernel for the Schaefer parcellation.

Writes ``schaefer{N}_geodesic.npy`` into ``data_dir`` -- an ``N x N`` matrix of
along-the-cortical-surface distances in millimetres, laid out LH-then-RH to
match ``schaefer{N}_centroids.csv``. Geodesic distance is undefined between
hemispheres, so the two off-diagonal blocks are ``nan``; slicing the first
``hidden_size`` rows/columns (what :func:`src.utils.load_embedding` does) yields
the finite left-hemisphere block the analyses use.

Distances are computed exactly, with the Mitchell-Mount-Papadimitriou algorithm
on the fsaverage white surface, rather than by shortest paths on the mesh graph.
Graph shortest paths are cheap but biased: restricted to mesh edges they can
only step in ~6 directions, so a geodesic that does not align with an edge is
approximated by a zigzag that overestimates length by ~7%. That bias is set by
the mesh's angular resolution, not its spatial resolution, so it does not shrink
with refinement -- it measures ~7% at fsaverage4 and ~7% at fsaverage7 alike.
Adding 2-ring/3-ring chords widens the direction set but lets paths short-circuit
across sulcal banks instead. Exact geodesics avoid both failure modes, and at a
one-off ~7 min for fsaverage7 there is no reason to accept either.

Requires ``pygeodesic``, which is a *build-time* dependency only -- nothing at
training or analysis time imports it, so it is deliberately absent from
requirements.txt. Install it just to regenerate this file::

    pip install --no-deps pygeodesic   # --no-deps keeps the pinned stack intact

Usage::

    python scripts/build_geodesic_kernel.py                  # 200 parcels, fsaverage7
    python scripts/build_geodesic_kernel.py --n-parcels 400
"""
import os
import argparse

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import nibabel as nib
from nilearn.datasets import load_fsaverage

from src.config import get_paths
from src.plotting import _get_schaefer_annot


def _cortex_submesh(coords, faces, labels):
    """Drop the medial wall, then keep the largest connected component.

    Both steps matter. Leaving the medial wall in lets paths cut across the
    callosal region, which is not cortex. Removing it can strand isolated
    vertices (5 per hemisphere at fsaverage7) that no path can reach; their
    distances come back infinite and silently poison the parcel means.
    """
    cortex = labels > 0
    keep_face = cortex[faces].all(axis=1)
    reindex = np.full(len(coords), -1, dtype=np.int64)
    reindex[cortex] = np.arange(cortex.sum())
    coords, labels, faces = coords[cortex], labels[cortex], reindex[faces[keep_face]]

    n = len(coords)
    edges = np.unique(np.sort(np.vstack(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1), axis=0)
    adj = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    _, component = connected_components(adj + adj.T, directed=False)
    main = component == np.bincount(component).argmax()
    if not main.all():
        print(f'      dropped {(~main).sum()} vertices stranded off the main component')
        keep_face = main[faces].all(axis=1)
        reindex = np.full(n, -1, dtype=np.int64)
        reindex[main] = np.arange(main.sum())
        coords, labels, faces = coords[main], labels[main], reindex[faces[keep_face]]

    return coords, faces, labels


def _parcel_seeds(coords, labels, n_parcels):
    """One seed vertex per parcel: the parcel vertex nearest its own centroid."""
    seeds = np.empty(n_parcels, dtype=np.int64)
    for i in range(n_parcels):
        v = np.flatnonzero(labels == i + 1)
        if v.size == 0:
            raise RuntimeError(f'parcel {i + 1} has no vertices on the cortical surface')
        seeds[i] = v[np.argmin(np.linalg.norm(coords[v] - coords[v].mean(axis=0), axis=1))]
    return seeds


def hemisphere_geodesic(coords, faces, labels, n_parcels, definition='mean'):
    """Exact geodesic distances between the parcels of one hemisphere.

    Args:
        coords, faces: white-surface mesh, full hemisphere.
        labels: per-vertex parcel index, 0 for the medial wall.
        n_parcels: parcels in this hemisphere.
        definition: how a parcel pair's distance is read off the distance
            field. ``'mean'`` averages over every vertex of the target parcel;
            ``'point'`` takes the target's seed vertex alone. ``'mean'`` is the
            default because it barely moves when the mesh is resampled, whereas
            ``'point'`` inherits the arbitrariness of which vertex happened to
            land nearest the centroid -- across fsaverage5/6/7 that choice
            shifts the seed ~1.2 mm and moves worst-case pair distances by 25%,
            against 10% for ``'mean'``.

    Returns:
        (n_parcels, n_parcels) symmetric array, zero diagonal, millimetres.
    """
    import pygeodesic.geodesic as geodesic

    coords, faces, labels = _cortex_submesh(coords, faces, labels)
    seeds = _parcel_seeds(coords, labels, n_parcels)

    algorithm = geodesic.PyGeodesicAlgorithmExact(
        coords.astype(np.float64), faces.astype(np.int32))
    field = np.stack([algorithm.geodesicDistances(np.array([s]), None)[0] for s in seeds])
    if not np.isfinite(field).all():
        raise RuntimeError('non-finite geodesic distances on a connected surface')

    if definition == 'mean':
        masks = [labels == j + 1 for j in range(n_parcels)]
        dist = np.array([[field[i, m].mean() for m in masks] for i in range(n_parcels)])
    elif definition == 'point':
        dist = field[:, seeds]
    else:
        raise ValueError(f"definition must be 'mean' or 'point', got {definition!r}")

    dist = (dist + dist.T) / 2          # seed-to-parcel is asymmetric by construction
    np.fill_diagonal(dist, 0.0)         # normalize_x/squareform require a zero diagonal
    return dist


def build_geodesic_matrix(n_parcels=200, mesh='fsaverage7', surface='white_matter',
                          definition='mean', yeo_networks=7):
    """Assemble the bilateral geodesic distance matrix, cross-hemisphere = nan."""
    per_hemi = n_parcels // 2
    fsaverage = load_fsaverage(mesh)
    lh_annot, rh_annot = _get_schaefer_annot(n_parcels, yeo_networks, mesh=mesh)

    out = np.full((n_parcels, n_parcels), np.nan)
    for i, (hemi, part, annot) in enumerate((('LH', 'left', lh_annot),
                                             ('RH', 'right', rh_annot))):
        print(f'    {hemi}: exact geodesics from {per_hemi} parcel seeds ...')
        part_mesh = fsaverage[surface].parts[part]
        block = hemisphere_geodesic(
            np.asarray(part_mesh.coordinates, dtype=np.float64),
            np.asarray(part_mesh.faces, dtype=np.int32),
            nib.freesurfer.read_annot(annot)[0],
            per_hemi, definition)
        s = slice(i * per_hemi, (i + 1) * per_hemi)
        out[s, s] = block
        off = block[np.triu_indices(per_hemi, 1)]
        print(f'      mean {off.mean():.1f} mm, max {off.max():.1f} mm')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-parcels', type=int, default=200,
                        help='bilateral parcel count (default: 200)')
    parser.add_argument('--mesh', default='fsaverage7',
                        help='fsaverage mesh to compute on (default: fsaverage7)')
    parser.add_argument('--surface', default='white_matter',
                        help='nilearn surface key (default: white_matter)')
    parser.add_argument('--definition', default='mean', choices=('mean', 'point'),
                        help='parcel-pair distance definition (default: mean)')
    parser.add_argument('--outdir', default=None,
                        help='output directory (default: data_dir from paths.yaml)')
    args = parser.parse_args()

    outdir = args.outdir or get_paths().data_dir
    print(f'Building geodesic kernel: {args.n_parcels} parcels on {args.mesh} '
          f'{args.surface}, {args.definition!r} definition')
    matrix = build_geodesic_matrix(args.n_parcels, args.mesh, args.surface, args.definition)

    fname = os.path.join(outdir, f'schaefer{args.n_parcels}_geodesic.npy')
    np.save(fname, matrix)
    print(f'  wrote {fname}  shape={matrix.shape} dtype={matrix.dtype}')


if __name__ == '__main__':
    main()
