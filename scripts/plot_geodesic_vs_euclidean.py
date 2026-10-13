"""Compare the geodesic and Euclidean spatial kernels for the left hemisphere.

Two panels over the 4,950 LH parcel pairs of the Schaefer-200 atlas:

* geodesic against Euclidean distance, with the identity line, to show the two
  kernels are not a rescaling of one another;
* their ratio against Euclidean distance, which localizes the disagreement --
  it is largest and most variable at short range, where parcels sit on opposite
  banks of a sulcus (close through the volume, far along the surface). That is
  also where a distance-penalty kernel puts most of its weight, so it is the
  regime in which swapping the metric actually changes what a model learns.

``--normalization`` replays what :func:`src.utils.load_embedding` does to each
matrix before training sees it, so the panels show the kernels as the
regularizer actually experiences them rather than in millimetres. Under
``mean`` both are divided by their own mean distance, which removes the ~1.47x
global scale difference and leaves only the disagreement in *shape* -- the part
a mean-normalized penalty can actually act on.

Reads ``schaefer{N}_geodesic.npy`` (see ``build_geodesic_kernel.py``) and
``schaefer{N}_centroids.csv`` from ``data_dir``. Usage::

    python scripts/plot_geodesic_vs_euclidean.py
    python scripts/plot_geodesic_vs_euclidean.py --normalization mean
    python scripts/plot_geodesic_vs_euclidean.py --outdir . --format pdf
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance as sp_distance
from scipy import stats
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from src.config import get_paths
from src.utils import get_my_colors, normalize_x


def load_distances(datadir, n_parcels=200, normalization='none'):
    """Geodesic and Euclidean distances over the left-hemisphere parcels.

    ``normalization`` is any :func:`src.utils.normalize_x` method, applied to
    each matrix exactly as ``load_embedding`` applies it, or ``'none'`` to keep
    raw millimetres.
    """
    per_hemi = n_parcels // 2
    geodesic = np.load(os.path.join(datadir, f'schaefer{n_parcels}_geodesic.npy'))
    geodesic = geodesic[:per_hemi, :][:, :per_hemi]
    if not np.isfinite(geodesic).all():
        raise ValueError('left-hemisphere geodesic block contains non-finite values')

    centroids = pd.read_csv(
        os.path.join(datadir, f'schaefer{n_parcels}_centroids.csv'))[:per_hemi]
    euclidean = sp_distance.squareform(
        sp_distance.pdist(centroids.set_index('ROI Name'), 'euclidean'))

    if normalization != 'none':
        geodesic = normalize_x(geodesic, normalization)
        euclidean = normalize_x(euclidean, normalization)

    upper = np.triu_indices(per_hemi, 1)
    return geodesic[upper], euclidean[upper]


def plot(geodesic, euclidean, n_parcels=200, normalization='none'):
    colors = get_my_colors()
    blue, raspberry, grey = (colors['starry_night_blue'],
                             colors['raspberry_blush'], (0.45, 0.45, 0.45))
    ratio = geodesic / euclidean
    r_pearson = stats.pearsonr(geodesic, euclidean)[0]
    r_spearman = stats.spearmanr(geodesic, euclidean)[0]
    if normalization == 'none':
        unit = ' (mm)'
        x_unit = ' (MNI centroids, mm)'
    else:
        unit = f' ({normalization}-normalized)'
        x_unit = f' (MNI centroids, {normalization}-normalized)'

    # binned medians: one summary curve per panel, so the trend is readable
    # through the overplotting without smoothing the raw points away
    edges = np.linspace(euclidean.min(), euclidean.max(), 19)
    which = np.digitize(euclidean, edges) - 1
    keep = [k for k in range(len(edges) - 1) if (which == k).sum() > 15]
    bin_x = np.array([euclidean[which == k].mean() for k in keep])
    bin_geodesic = np.array([np.median(geodesic[which == k]) for k in keep])
    bin_ratio = np.array([np.median(ratio[which == k]) for k in keep])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), dpi=220)
    lim = (0, max(euclidean.max(), geodesic.max()) * 1.04)
    label_pad = 0.032 * (lim[1] - lim[0])

    ax = axes[0]
    ax.plot(lim, lim, color=grey, lw=1.2, ls=(0, (4, 3)), zorder=1)
    # place the label out past the data, not on top of it: once the scale
    # difference is normalized away the cloud sits directly on this line
    ax.text(lim[1] * 0.88, lim[1] * 0.88, 'identity', color=grey, fontsize=7,
            rotation=45, rotation_mode='anchor', va='bottom', ha='center')
    ax.scatter(euclidean, geodesic, s=5, color=blue, alpha=0.18, lw=0,
               zorder=2, rasterized=True)
    ax.plot(bin_x, bin_geodesic, color=raspberry, lw=2, zorder=3, solid_capstyle='round')
    ax.text(bin_x[-1], bin_geodesic[-1] + label_pad, 'binned median', color=raspberry,
            fontsize=7, ha='right', va='bottom')
    ax.set_xlabel('Euclidean distance' + x_unit)
    ax.set_ylabel('Geodesic distance' + unit)
    ax.set_xlim(lim), ax.set_ylim(lim), ax.set_aspect('equal')
    ax.set_title(f'$r$ = {r_pearson:.3f}   $\\rho$ = {r_spearman:.3f}',
                 fontsize=8, color='0.25', pad=6)

    ax = axes[1]
    ax.axhline(1, color=grey, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.scatter(euclidean, ratio, s=5, color=blue, alpha=0.18, lw=0,
               zorder=2, rasterized=True)
    ax.plot(bin_x, bin_ratio, color=raspberry, lw=2, zorder=3, solid_capstyle='round')
    ax.set_xlabel('Euclidean distance' + x_unit)
    ax.set_ylabel('Geodesic / Euclidean')
    # same x-limits as the left panel: both encode Euclidean distance, so
    # reading across the two is only honest if the axis matches
    ax.set_xlim(lim)
    # log y: a ratio is multiplicative, so 2x and 0.5x are equal-sized
    # disagreements and must occupy equal vertical distance. On a linear axis
    # the "geodesic longer" half gets several times the space of the "geodesic
    # shorter" half, which badly misreads the normalized kernels -- once the
    # global scale is divided out, roughly half the pairs fall below 1.
    ax.set_yscale('log')
    ax.set_ylim(ratio.min() / 1.06, ratio.max() * 1.06)
    ticks = np.array([0.25, 1 / 3, 0.5, 2 / 3, 0.8, 1, 1.25, 1.5, 2, 3, 4, 5, 6])
    ticks = ticks[(ticks >= ratio.min() / 1.06) & (ticks <= ratio.max() * 1.06)]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f'{v:g}' if v >= 1 else f'{v:.2f}'.rstrip('0').rstrip('.')))
    ax.set_title(f'median {np.median(ratio):.2f}×   max {ratio.max():.1f}×',
                 fontsize=8, color='0.25', pad=6)

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('0.55')
        ax.tick_params(colors='0.35', labelsize=7.5, length=3)
        ax.xaxis.label.set_size(8), ax.yaxis.label.set_size(8)
        ax.grid(True, color='0.9', lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    norm_note = '' if normalization == 'none' else f', {normalization}-normalized'
    fig.suptitle('Geodesic vs Euclidean inter-parcel distance — '
                 f'{n_parcels // 2} LH Schaefer-{n_parcels} parcels{norm_note}',
                 fontsize=9, y=1.0)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-parcels', type=int, default=200,
                        help='bilateral parcel count (default: 200)')
    parser.add_argument('--datadir', default=None,
                        help='input directory (default: data_dir from paths.yaml)')
    parser.add_argument('--outdir', default=None,
                        help='output directory (default: figure_dir from paths.yaml)')
    parser.add_argument('--format', default='png', help='figure format (default: png)')
    parser.add_argument('--normalization', default='none',
                        choices=('none', 'rescale', 'mean', 'meansq', 'mean_std', 'uniform'),
                        help='normalize_x method applied to both matrices, matching '
                             'load_embedding (default: none, i.e. raw mm)')
    args = parser.parse_args()

    paths = get_paths()
    datadir = args.datadir or paths.data_dir
    outdir = args.outdir or paths.figure_dir
    os.makedirs(outdir, exist_ok=True)

    geodesic, euclidean = load_distances(datadir, args.n_parcels, args.normalization)
    ratio = geodesic / euclidean
    print(f'{len(geodesic)} LH parcel pairs, normalization={args.normalization!r}')
    u = 'mm' if args.normalization == 'none' else 'a.u.'
    print(f'  euclidean  mean {euclidean.mean():6.3f}  range {euclidean.min():6.3f}-{euclidean.max():6.3f} {u}')
    print(f'  geodesic   mean {geodesic.mean():6.3f}  range {geodesic.min():6.3f}-{geodesic.max():6.3f} {u}')
    print(f'  Pearson r {stats.pearsonr(geodesic, euclidean)[0]:.4f}   '
          f'Spearman rho {stats.spearmanr(geodesic, euclidean)[0]:.4f}')
    print(f'  ratio: median {np.median(ratio):.3f}  '
          f'IQR {np.percentile(ratio, 25):.3f}-{np.percentile(ratio, 75):.3f}  max {ratio.max():.2f}')

    fig = plot(geodesic, euclidean, args.n_parcels, args.normalization)
    suffix = '' if args.normalization == 'none' else f'_{args.normalization}norm'
    fname = os.path.join(
        outdir, f'geodesic_vs_euclidean_schaefer{args.n_parcels}{suffix}.{args.format}')
    fig.savefig(fname, bbox_inches='tight', facecolor='white')
    print(f'\nFigure saved: {fname}')


if __name__ == '__main__':
    main()
