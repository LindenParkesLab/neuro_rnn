# data_private

**Everything in this folder is git-ignored** (except this README). Put data here
that must not be redistributed.

This is where the empirical fMRI inputs belong. They are **not** included in this
repository: the Human Connectome Project (HCP) requires users to register and
accept its Data Use Terms, so we cannot ship them.

## What goes here

Point `fmri_dir` in your `paths.yaml` at this folder (the shipped
`paths.yaml.template` already does), then place the following files here:

| File | Contents |
|---|---|
| `hcpya_tfmri.pkl` | HCP-YA task fMRI (N-back / working memory), parcellated |
| `HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest.npy` | HCP-YA resting-state time series |
| `HCP_YA_Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_rest_df.csv` | Subject table for the resting-state data |

Time series are parcellated with the Schaefer-200 (7-network) atlas; only the
100 left-hemisphere parcels are used, to match the RNN hidden layer.

## Obtaining the data

HCP Young Adult data are available from the
[HCP data release](https://www.humanconnectome.org/study/hcp-young-adult) via
[ConnectomeDB](https://db.humanconnectome.org). Access requires accepting the
HCP Open Access Data Use Terms.

Analyses that do not read empirical fMRI (e.g. RNN training and task
performance) run without any of this.

## Why this folder exists

Openly redistributable inputs live in [`../data_public`](../data_public) and
are tracked in git. Restricted data lives here and never is. Keeping the two
apart means a stray file cannot be committed by accident.
