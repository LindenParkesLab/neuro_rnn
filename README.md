# neuro_rnn

## Background

Recurrent neural networks (RNNs) are *in silico* networks capable of reading in time series data and producing time series outputs. They are useful models to study simulated brain dynamics and the links between brain structure and function. Classically, RNNs are trained without spatial constraints on their hidden weights. However, the brain is a physically embedded system with clear structural constraints on its shape and connectivity. Therefore, this repository supports training and analyzing **biophysical RNNs (bioRNNs)**, a class of RNNs that have such constraints built into their architecture.

---

## Reference paper
Beyh A., Kim J.Z., Bajwa W.U., & Parkes L. *Geometric constraints and cognitive inputs jointly shape emergent brain dynamics and topology.* [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.08.07.742341)

---

## Requirements

- Linux or macOS
- [conda](https://docs.conda.io/) (Miniconda or Anaconda)
- Python 3.12.11 and PyTorch 2.7.1 — installed by `environment.yml` (see below)
- Optional: an NVIDIA GPU for CUDA training

The published results were produced on a Mac Studio (M3 Ultra) using CPU.

---

## Installation

The code requires several dependencies, and specific versions must be used to avoid clashes between them. This can be achieved in a few steps that should guarantee you get the exact setup needed. **Important:** make sure you have conda installed before proceeding.

### 1. Clone the main code

First, clone the code into a local repository on your machine.

```bash
git clone https://github.com/LindenParkesLab/neuro_rnn.git
cd neuro_rnn
```

### 2. Create the conda environment

From within the `neuro_rnn` repo, create a new conda environment using the existing `environment.yml` config file and activate it.

```bash
conda env create -f environment.yml
conda activate neuro_rnn
```

### 3. Install the local package

Now that the installation configuration is in place, install the main package. This step will automatically download and install the correct version of each required dependency. 

```bash
pip install -e .
```

### 4. Configure your paths

Running the code requires setting up data paths. Copy the existing template, `paths.yaml.template`, and edit it to match your local paths. See **Configuration** below for more details.

```bash
cp paths.yaml.template paths.yaml
```

### 5. Verify the installation

Run this quick python command to verify that the installation worked.

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
| `fmri_dir` | Restricted empirical fMRI inputs | `./data_private` |
| `figure_dir` | Where analyses save figures | `<model_dir>/figures` |

Relative paths resolve against the repository root, so they behave identically whether you run from the repo root, from `scripts/`, or inside a notebook. Any key can be overridden at run time with `NEURO_RNN_DATA_DIR`, `NEURO_RNN_MODEL_DIR`, `NEURO_RNN_FMRI_DIR`, or `NEURO_RNN_FIGURE_DIR`.

`model_dir` and `figure_dir` gain a subdirectory named after the params CSV (training options), so different training setups never collide. 

In code:

```python
from src.config import get_paths, ensure_dir

paths  = get_paths('model_params_202606d', require='all')
figdir = ensure_dir(paths.figure_dir)
```

---

## Data

### `data_public/` — ships with the repo

Atlas geometry, regularization kernels, and the model parameter files. These cover everything required for training the models. See [`data_public/README.md`](data_public/README.md) for a file-by-file description and sources.

Atlas files are named by the **bilateral** parcel count and selected as `hidden_size * 2` — so `hidden_size=100` (the 100 left-hemisphere parcels used throughout the paper) reads the `schaefer200_*` files.

### `data_private/` — you supply this

The empirical fMRI inputs. **Not included**: the HCP requires users to register and accept its Data Use Terms. Everything in this folder is git-ignored, so restricted data cannot be committed by accident. See [`data_private/README.md`](data_private/README.md) for the expected files and how to obtain them.

---

## Quick test

Verify the pipeline end-to-end with the test parameters file. It defines one model
(`n_runs=4`, `n_epochs=1000`), which is enough to test all the training functionality without a long wait (the resulting models are not meaningful).

```bash
python scripts/train_rnn.py --params_csv data_public/model_params_test.csv --model_index 0
```

On success you'll see training progress and three files (`*_models.h5`, `*_outputs.h5`, `*_config.npy`) under `<model_dir>/model_params_test/`.

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
| `rnn_model` | RNN non-linearity | `rnn-tanh` |
| `hidden_size` | number of hidden units (parcels) | `100` |
| `batch_size`, `learning_rate`, `n_epochs` | optimizer settings | `32`, `0.001`, `60000` |
| `reg_type` | `l1`, `l2`, `l2s`, `pearson`, `pearson_l2s`, `pearson_abs`, `pearson_abs_l2s` | `pearson_l2s` |
| `reg_weight` | regularization weight (λ) | `0.002` |
| `kernel_type` | spatial kernel (`None`, `euclidean`, `sa_axis`, `struct_conn`, …) | `euclidean` |
| `kernel_normalization` | kernel normalization | `mean` |
| `mask_weights` | apply projection constraints (read-in / read-out masks) | `True` |
| `n_runs` | number of randomly initialized networks | `100` |
| `null_perms`, `null_seed`, `null_run`, `null_run_ids` | spin-permutation controls for the geometry null | `200`, `0`, `0`, `14 24 32 57 87` |

### The published results

`data_public/model_params_202606d.csv` reproduces the paper:

| Row | `reg_type` | `kernel_type` | `mask_weights` | Runs | Corresponds to |
|---|---|---|---|---|---|
| 0 | `l2s` | – | `False` | 100 | **Vanilla RNNs** |
| 1 | `l2s` | – | `True` | 100 | **Masked RNNs** |
| 2 | `pearson_l2s` | `euclidean` | `True` | 100 | **bioRNNs** |
| 3 | `pearson_l2s` | `euclidean` | `True` | 5 × 200 perms | the 1,000 geometry-null RNNs |

Row 3's `null_run_ids` (`14 24 32 57 87`) are the five run indices reused across every permuted geometry, so any difference between geometries is attributable to geometry alone.

**Compute warning:** rows 0–2 are 100 networks × 60,000 epochs each, and row 3 is 1,000 networks. This is many CPU-days of compute. Start with the quick test.

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
  frequencies in epochs (defaults: `100`, `100`, `1000`)

Any CSV column can be overridden on the command line (e.g. `--n_epochs`, `--reg_type`);
command-line values take precedence over the CSV.

#### Frequency constraints

The three frequencies must divide the epoch count, and writes must align with logs:

```
n_epochs   % print_freq == 0
n_epochs   % log_freq   == 0
n_epochs   % write_freq == 0
write_freq % log_freq   == 0
```

Violations raise a `ValueError` listing each offending pair before training starts.

These rules keep checkpoints aligned with the end of training: analyses read the
**final** checkpoint (epoch 40,000 in the paper), and the training-trajectory analysis
assumes evenly spaced checkpoints across the run. If `n_epochs` were not a multiple of
`write_freq`, the last partial window would never be written and the final state would
be missing.

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

The launcher fixes the frequencies at `print_freq=100`, `log_freq=100`,
`write_freq=1000` (edit the script to change them), so every row of the CSV it runs
must have an `n_epochs` that is a multiple of 1,000 — see
[Frequency constraints](#frequency-constraints) above.

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

## Reproducing the figures

Four notebooks produce every figure in the paper. Each loads its inputs through
`paths.yaml`, so once that is configured they can be run in any order — with one
exception noted below.

| Notebook | Figures | Needs |
|---|---|---|
| `biornn_analysis_performance` | 1e, 1f, S1, S2 | models |
| `biornn_analysis_dynamics` | 2b, 2c, 2d, 2e, S3, S4 | models + fMRI |
| `biornn_analysis_trajectory` | 3a, 3b, S5, S9 | models + fMRI + trajectory results |
| `biornn_analysis_topology` | 4, S6, S7, S8 | models + fMRI + trajectory results |

**Without HCP access**, `biornn_analysis_performance` still runs end to end: task
performance, learning speed and the loss terms need only the trained networks. The other
three compare network dynamics against empirical fMRI and cannot run without it.

**Trajectory results** are the pickle written by
`scripts/biornn_results_dynamics_trajectory.py` (see below). Produce it once; the
trajectory and topology notebooks both read it rather than recomputing.

Figures are written to `figure_dir` from `paths.yaml`, in a subdirectory named after the
params CSV, and named for the paper figure they produce (`fig2c_...svg`, `figS4_...svg`).

### What the notebooks share

Analysis code lives in `src/`, not in the notebooks, so the figures are built from one
implementation rather than several:

- `src/performance.py` — learning curves, convergence epochs, paired class statistics
- `src/dynamics.py` — evaluating trained runs, hidden-state PCA, weight–geometry similarity
- `src/trajectory.py` — reading the trajectory results, training-phase onsets
- `src/pca_utils.py` — PCA and subspace-variance machinery
- `src/null_utils.py` — the random-subspace and spin-permutation nulls
- `src/topology.py` — graph metrics and connection length

Two consequences worth knowing. **Variance explained is reported as a z-score against a
random-subspace null** throughout: a five-dimensional subspace captures a non-trivial
share of fMRI variance purely by virtue of its dimensionality, so raw values cannot be
read on their own, and z ≈ 0 is chance. Every figure computes it through the same
function, so the scales are comparable. And **evaluation is seeded**: the task battery
and noise drive are random draws, so an unseeded run would shift every downstream value
between executions.

### The analysis epoch

Networks are evaluated at the point where every class has reached stable performance.
That epoch is *derived* from the learning curves rather than hard-coded — the performance
notebook fits each run's accuracy trajectory, takes the slowest class's criterion, and
rounds up. For the published sweep it works out to 40,000.

---

## Analyses

These require trained models, and (except where noted) empirical fMRI.

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

## Note for contributors

If you plan to commit changes, register the `nbstripout` filter once after cloning:

```bash
nbstripout --install --attributes .gitattributes
```

Notebook outputs are then stripped automatically when staging, while your local copies
keep their figures. This keeps the repository small and stops local paths from leaking
into version control through embedded cell outputs. The filter is stored in `.git/config`,
so it is per-clone and needs registering on each machine you commit from.

---

## Citation

If you use this code, please cite the paper above.

## License

BSD 3-Clause — see [LICENSE](LICENSE).
