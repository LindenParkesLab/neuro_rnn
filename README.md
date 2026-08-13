# neuro_rnn

Training and analysis code for **biophysical RNNs (bioRNNs)** — spatially embedded,
biologically constrained recurrent neural networks used to study how the brain's
physical geometry and cognitive inputs jointly shape emergent dynamics and topology.

Three RNN classes form a graded hierarchy of spatial constraint:

| Class | Projection constraints | Geometry prior on recurrent weights |
|---|---|---|
| **Vanilla** | – | – |
| **Masked** | ✓ | – |
| **bioRNN** | ✓ | ✓ (inter-regional Euclidean distance) |

Networks are trained on a delayed match-to-sample with distractors (DMS-D) task via
[NeuroGym](https://github.com/neurogym/neurogym), and their hidden-state dynamics are
compared against empirical HCP task fMRI.

> **Paper:** Beyh, A., Kim, J. Z., Bajwa, W. U., & Parkes, L. *Geometric constraints and
> cognitive inputs jointly shape emergent brain dynamics and topology.*

---

## Requirements

- Linux or macOS (Apple silicon supported via `mps`)
- [conda](https://docs.conda.io/) (Miniconda or Anaconda)
- Python 3.12.11 and PyTorch 2.7.1 — installed by `environment.yml`
- Optional: an NVIDIA GPU for CUDA training

The published results were produced on a Mac Studio (M3 Ultra) using CPU/`mps`.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/abeyh/neuro_rnn.git
cd neuro_rnn
```

### 2. Create the environment

`environment.yml` installs Python and defers every library to `requirements.txt`,
where all dependencies are pinned to the exact versions used for the paper.

```bash
conda env create -f environment.yml
conda activate neuro_rnn
```

> Conda supplies only the interpreter; pip resolves the rest. This is deliberate —
> NeuroGym hard-pins `numpy`, `scipy` and `matplotlib`, and letting conda and pip each
> resolve part of the stack is a common source of version clashes.

> **GPU:** the pinned `torch` is the default CPU/MPS wheel. For an NVIDIA GPU, install
> the matching CUDA build of torch 2.7.1 afterwards, per
> [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Install the local package

```bash
pip install -e .
```

### 4. Configure your paths

Copy the template and edit it. `paths.yaml` is git-ignored, so your local paths never
enter version control.

```bash
cp paths.yaml.template paths.yaml
```

### 5. Verify

```bash
python -c "import torch, neurogym, bct; from src.config import get_paths; print('OK', torch.__version__)"
```

---

## Configuration

All directory locations live in one file, `paths.yaml`:

| Key | Meaning | Default |
|---|---|---|
| `data_dir` | Openly redistributable inputs (ships with the repo) | `./data_public` |
| `model_dir` | Where trained models are written and read | `./results/model` |
| `fmri_dir` | Restricted HCP fMRI inputs | `./data_private` |
| `figure_dir` | Where analyses save figures | `<model_dir>/figures` |

Relative paths resolve against the repository root, so they behave identically whether
you run from the repo root, from `scripts/`, or inside a notebook. Any key can be
overridden at run time with `NEURO_RNN_DATA_DIR`, `NEURO_RNN_MODEL_DIR`,
`NEURO_RNN_FMRI_DIR`, or `NEURO_RNN_FIGURE_DIR`.

`model_dir` and `figure_dir` gain a subdirectory named after the params CSV, so
different training sweeps never collide. Point `model_dir` at a disk with room to
spare — a full 100-run sweep with checkpoints is large.

In code:

```python
from src.config import get_paths, ensure_dir

paths  = get_paths('model_params_202606d', require='all')
figdir = ensure_dir(paths.figure_dir)
```

---

## Data

### `data_public/` — ships with the repo

Atlas geometry, regularization kernels, and the model parameter files. Everything
training needs. See [`data_public/README.md`](data_public/README.md) for a
file-by-file description and sources.

Atlas files are named by the **bilateral** parcel count and selected as
`hidden_size * 2` — so `hidden_size=100` (the 100 left-hemisphere parcels used
throughout the paper) reads the `schaefer200_*` files.

### `data_private/` — you supply this

The empirical HCP fMRI inputs. **Not included**: the Human Connectome Project requires
users to register and accept its Data Use Terms. Everything in this folder is
git-ignored, so restricted data cannot be committed by accident. See
[`data_private/README.md`](data_private/README.md) for the expected files and how to
obtain them.

RNN training and task-performance analyses run **without** any of this; only the
fMRI-comparison analyses need it.

---

## Quick test

Verify the pipeline end-to-end with the bundled test config. It defines one model
(`n_runs=4`, `n_epochs=1000`) — enough to exercise parallel runs, checkpointing, and
logging without a long wait:

```bash
python scripts/train_rnn.py --params_csv data_public/model_params_test.csv --model_index 0
```

Paths come from `paths.yaml`, so no directory arguments are needed. On success you'll
see training progress and three files (`*_models.h5`, `*_outputs.h5`, `*_config.npy`)
under `<model_dir>/model_params_test/`.

---

## Training

### The parameters CSV

Each **row** defines one model configuration; rows are 0-indexed (the header is not
counted). Empty cells fall back to code defaults.

| column | meaning | example |
|---|---|---|
| `task` | neurogym task id | `DelayMatchSampleDistractor1D-v0` |
| `seq_len_multi`, `time_step` | sequence length multiplier, ms per step | `5`, `50` |
| `alpha` | temporal integration factor (float, or a `.txt` filename in `data_public/` with one value per node) | `0.1` |
| `rec_noise` | recurrent noise | `0.05` |
| `init_hh_w` | hidden-hidden weight init (`float`, `"min max"`, or `None`) | `-0.01 0.01` |
| `rnn_model` | RNN cell type | `rnn-tanh` |
| `hidden_size` | number of hidden units (parcels) | `100` |
| `batch_size`, `learning_rate`, `n_epochs` | optimizer settings | `32`, `0.001`, `60000` |
| `reg_type` | `l1`, `l2`, `l2s`, `pearson`, `pearson_l2s`, `pearson_abs`, `pearson_abs_l2s` | `pearson_l2s` |
| `reg_weight` | regularization weight (λ) | `0.002` |
| `kernel_type` | spatial kernel (`None`, `euclidean`, `sa_axis`, `struct_conn`, …) | `euclidean` |
| `kernel_normalization` | kernel normalization | `mean` |
| `mask_weights` | apply projection constraints (read-in / read-out masks) | `True` |
| `n_runs` | number of randomly initialized networks | `100` |
| `null_perms`, `null_seed`, `null_run`, `null_run_ids` | spin-permutation controls for the geometry null | `200`, `0`, `0`, `14 24 32 57 87` |

### The published sweep

`data_public/model_params_202606d.csv` reproduces the paper:

| Row | `reg_type` | `kernel_type` | `mask_weights` | Runs | Corresponds to |
|---|---|---|---|---|---|
| 0 | `l2s` | – | `False` | 100 | **Vanilla RNNs** |
| 1 | `l2s` | – | `True` | 100 | **Masked RNNs** |
| 2 | `pearson_l2s` | `euclidean` | `True` | 100 | **bioRNNs** |
| 3 | `pearson_l2s` | `euclidean` | `True` | 1 × 200 perms | geometry-null pilot |
| 4 | `pearson_l2s` | `euclidean` | `True` | 5 × 200 perms | the 1,000 geometry-null RNNs |

Row 4's `null_run_ids` (`14 24 32 57 87`) are the five run indices reused across every
permuted geometry, so any difference between geometries is attributable to geometry
alone.

> **Compute warning:** rows 0–2 are 100 networks × 60,000 epochs each, and row 4 is
> 1,000 networks. This is many CPU-days of compute. Start with the quick test.

### Option A — Python entry point

```bash
python scripts/train_rnn.py \
  --params_csv data_public/model_params_202606d.csv \
  --model_index 2 \
  --device cpu \
  --n_threads 4
```

Key arguments:

- `--params_csv` / `--model_index` — the CSV and which row (must be given together)
- `--datadir` / `--outdir` — optional; default to `data_dir` and
  `<model_dir>/<params_csv name>` from `paths.yaml`
- `--device` — `cpu` (default), `cuda`, `mps`, or `gpu` (auto-selects `cuda`/`mps` if
  available). Accelerators are never chosen implicitly, so a run does not silently
  change hardware depending on the machine.
- `--n_threads` — CPU threads
- `--print_freq` / `--log_freq` / `--write_freq` — console, metric, and checkpoint
  frequencies in epochs

Any CSV column can be overridden on the command line (e.g. `--n_epochs`, `--reg_type`);
command-line values take precedence over the CSV.

### Option B — bash launcher

Iterates rows of a CSV, creating the output directory and writing a timestamped log per
model index:

```bash
bash scripts/train_rnn.sh data_public/model_params_202606d.csv        # all rows
bash scripts/train_rnn.sh data_public/model_params_202606d.csv 2      # row 2 only
bash scripts/train_rnn.sh data_public/model_params_202606d.csv 0 1 2  # the three classes
```

Paths come from `paths.yaml`; nothing in the script needs editing. Device and threads
can be set via the environment:

```bash
DEVICE=mps N_THREADS=8 bash scripts/train_rnn.sh data_public/model_params_202606d.csv 2
```

Set `SKIP_CONDA_ACTIVATE=1` if you manage the environment yourself.

### Outputs

Three files per trained model, named from the config:

- `<file_str>_models.h5` — trained weights / checkpoints
- `<file_str>_outputs.h5` — training metrics (losses, activations)
- `<file_str>_config.npy` — the full config used

where `<file_str>` encodes the configuration, e.g.:

```
DelayMatchSampleDistractor1D-v0-520-rnn-tanh-100-32-0.001-100-60000-True-14-27-pearson_l2s-0.002-euclidean-mean-0.1
```

---

## Analyses

These require trained models, and (except where noted) empirical fMRI in `data_private/`.

**`scripts/biornn_results_dynamics_trajectory.py`** — the main analysis engine. Computes
weight–kernel similarity, fMRI variance explained, and network topology across training
checkpoints, saving results as a pickle for downstream plotting:

```bash
python scripts/biornn_results_dynamics_trajectory.py \
  --model_params model_params_202606d \
  --rows 2 \
  --n_epochs 10 \
  --n_trials 100 \
  --n_fmri_subj 100 \
  --n_pc 5 \
  --n_jobs -1
```

Use `--skip_fmri` to run without empirical data, and `--skip_topology` to skip graph
metrics. Run with `--help` for the full argument list.

**`scripts/spatial_null_precheck.py`** — an RNN-free check of whether Euclidean geometry
explains fMRI variance beyond spatial autocorrelation.

**`scripts/compute_rnn_topology.py`** — standalone graph-theoretic metrics over trained
models.

---

## Repository layout

```
neuro_rnn/
├── environment.yml          # conda environment (Python only)
├── requirements.txt         # exact dependency pins
├── setup.py
├── paths.yaml.template      # copy to paths.yaml and edit
├── data_public/             # atlas geometry, kernels, params CSVs (tracked)
├── data_private/            # restricted HCP fMRI (git-ignored; you supply)
├── scripts/
│   ├── train_rnn.py         # training entry point
│   ├── train_rnn.sh         # bash launcher over a params CSV
│   └── biornn_results_*.py  # analysis engines
└── src/
    ├── config.py            # path configuration (paths.yaml)
    ├── neural_network.py    # RNN model, training loop, state managers
    ├── io_utils.py          # config building + CSV parsing
    ├── utils.py             # task setup, device handling, kernels
    ├── fmri_io.py           # HCP fMRI loading, GSR, kernels
    ├── null_utils.py        # random-subspace and spin-null testing
    ├── spatial_null.py      # spin-permutation null geometry
    ├── topology.py          # graph metrics
    └── plotting.py          # figure helpers
```

---

## Citation

If you use this code, please cite the paper above.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
