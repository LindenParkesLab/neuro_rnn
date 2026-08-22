# data_private

**Everything in this folder is git-ignored** (except this README). Put data here
that must not be redistributed.

This is where the empirical fMRI inputs belong. They are **not** included in this
repository: the Human Connectome Project (HCP) requires users to register and
accept its Data Use Terms, so we cannot ship them.

## What goes here

The analyses expect these three files:

| File | Contents | Approx. size |
|---|---|---|
| `hcpya_tfmri.pkl` | HCP-YA task fMRI (N-back / working memory), parcellated | ~19 GB |
| `HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest.npy` | HCP-YA resting-state time series | ~9 GB |
| `HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest_df.csv` | Subject table for the resting-state data | small |

Time series are parcellated with the Schaefer-200 (7-network) atlas; only the
100 left-hemisphere parcels are used, to match the RNN hidden layer.

**You do not have to put them in this folder.** Given their size, keeping them on a
separate volume is usually preferable — point `fmri_dir` in your `paths.yaml` at
wherever they live:

```yaml
fmri_dir: /path/to/your/hcp/data
```

This folder is simply the default, and a safe place for restricted data if you do
want it alongside the code: its contents can never be committed.

## Obtaining the data

HCP Young Adult data are available from the
[HCP data release](https://www.humanconnectome.org/study/hcp-young-adult) via
[ConnectomeDB](https://db.humanconnectome.org). Access requires accepting the
HCP Open Access Data Use Terms.

Analyses that do not read empirical fMRI (e.g. RNN training and task
performance) run without any of this.

## Why this folder exists

Openly redistributable inputs live in [`../data_public`](../data_public) and are
tracked in git. Restricted data either lives here — where it can never be
committed — or outside the repository entirely, with `fmri_dir` pointing at it.
Keeping the two apart means a stray file cannot end up in version control by
accident.
