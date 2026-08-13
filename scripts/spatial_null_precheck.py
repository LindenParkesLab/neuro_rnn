"""Kernel-level pre-check: does the euclidean geometry explain fMRI variance
beyond its spatial autocorrelation?

RNN-free. Builds the euclidean similarity kernel, takes its top-k spatial
gradient eigenmodes as a fixed basis, and asks how much fMRI temporal variance
that basis captures -- versus an ensemble of spin (SA-matched) surrogates of the
same basis. If real <= spin distribution, the geometry is "just SA/smoothness"
(H1) and training surrogate models won't help; if real > spin, the specific
geometry matters (H2) and the surrogate-kernel training is worth it.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import distance

import src.utils as utils
from src.fmri_io import get_paths, load_fmri_data
from src.spatial_null import (
    spin_permutations, gradient_modes, basis_fmri_ve, spin_null_ve, spin_pvalue)

N_FMRI_SUBJ = 100
N_PERM = 1000
KS = [2, 3, 5, 10]
KERNEL_NORM = 'mean'           # match the bioRNN training kernel normalization
SEED = 0


def main():
    paths = get_paths('model_params_202606d', require='all')

    # euclidean kernel from the volumetric centroids (what the bioRNN trains on)
    centroids = pd.read_csv(os.path.join(paths.data_dir, 'schaefer200_centroids.csv'))
    centroids = centroids.set_index('ROI Name')[:100]
    D = distance.squareform(distance.pdist(centroids, 'euclidean'))
    S = 1 - utils.normalize_x(D, KERNEL_NORM)

    # spin permutations from the FreeSurfer-sphere coords of the same parcels
    sphere = np.loadtxt(os.path.join(paths.data_dir, 'schaefer200_LH_sphere_coords.txt'))
    perms = spin_permutations(sphere, N_PERM, seed=SEED)

    # fMRI (GSR'd, seeded common subjects) via the shared loader
    fmri_task_ts, fmri_rest_ts, _, _ = load_fmri_data(
        paths.data_dir, paths.fmri_dir, N_FMRI_SUBJ)

    print(f"\n{'k':>3} {'modality':>5} {'real_VE':>9} {'spin_mean':>10} "
          f"{'spin_p95':>9} {'p_spin':>8} {'verdict'}")
    results = {}
    for k in KS:
        U = gradient_modes(S, k)
        for label, ts in (('task', fmri_task_ts), ('rest', fmri_rest_ts)):
            real = basis_fmri_ve(ts, U)
            null = spin_null_ve(ts, U, perms)
            p = spin_pvalue(real, null)
            verdict = 'H2 (geometry)' if p < 0.05 else 'H1 (SA only)'
            print(f"{k:>3} {label:>5} {real:>9.4f} {null.mean():>10.4f} "
                  f"{np.percentile(null,95):>9.4f} {p:>8.4f}  {verdict}")
            results[(k, label)] = (real, null, p)

    # figure for the headline k = 5
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        utils.set_font_size(11)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for ax, label in zip(axes, ('task', 'rest')):
            real, null, p = results[(5, label)]
            ax.hist(null, bins=40, color='0.7', edgecolor='none')
            ax.axvline(real, color='crimson', lw=2,
                       label=f'real geometry\n(p={p:.3f})')
            ax.set_title(f'{label} fMRI  (k=5 gradients)')
            ax.set_xlabel('fMRI variance explained')
            ax.set_ylabel('# spin surrogates')
            ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        out = '/tmp/spatial_null_precheck.png'
        fig.savefig(out, dpi=130, bbox_inches='tight')
        print(f"\nFigure saved: {out}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == '__main__':
    main()
