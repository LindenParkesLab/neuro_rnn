# data_public

Openly redistributable inputs, **tracked in git** and shipped with the repo.
Everything training needs is here; nothing in this folder is restricted.

> Restricted data (HCP fMRI) does **not** belong here — it goes in
> [`../data_private`](../data_private), which is git-ignored.

Point `data_dir` in `paths.yaml` at this folder (the default already does).

## Model parameter files

Each row of a params CSV defines one model configuration; see the README's
"parameters CSV" section for the column schema.

| File | Purpose |
|---|---|
| `model_params_202606d.csv` | The sweep used for the published analyses |
| `model_params_202608a.csv` | Additional sweep |
| `model_params_test.csv` | Small config for the README quick test (4 runs, 1000 epochs) |

## Atlas geometry

Derived from the Schaefer 2018 local-global parcellation. Files are named by
the **bilateral** parcel count, and are selected as `hidden_size * 2` — so
`hidden_size=100` (the 100 left-hemisphere parcels used throughout the paper)
reads the `schaefer200_*` files.

| File | Contents |
|---|---|
| `schaefer{100,200,400}_centroids.csv` | Parcel centroid coordinates. The pairwise Euclidean distances between these define the bioRNN spatial embedding. |
| `schaefer{200,400}_{LH,RH}_sphere_coords.txt` | Parcel coordinates on the FreeSurfer spherical surface, used to generate spin permutations for the geometry null. |
| `schaefer200_spherical_euclidean.txt` | Euclidean distances computed on the spherical surface. |

## Regularization kernels and brain maps

Selected via the `kernel_type` column of a params CSV. The published analyses
use only the `euclidean` kernel (built from the centroids above); the remainder
support alternative embeddings and comparison analyses.

| File | Contents |
|---|---|
| `schaefer{100,200,400}_sa-axis.npy` | Sensorimotor–association axis. Used as the target in the cortical-hierarchy regression. |
| `schaefer200_sa-axis_alpha_lh.txt` | Left-hemisphere S–A axis values. |
| `schaefer200_ut-axis.npy` | Unimodal–transmodal axis. |
| `schaefer{100,200,400}_cyto.npy` | Cytoarchitectural map. |
| `schaefer200_myelin.npy` | Myelin map. |
| `schaefer200_structural_conn_kernel.npy` | Group-average structural connectivity matrix (200×200). |

## References

- Schaefer, A. et al. Local-global parcellation of the human cerebral cortex
  from intrinsic functional connectivity MRI. *Cerebral Cortex* **28**,
  3095–3114 (2018).
- Sydnor, V. J. et al. Neurodevelopment of the association cortices: patterns,
  mechanisms, and implications for psychopathology. *Neuron* **109**,
  2820–2846 (2021). — sensorimotor–association axis
