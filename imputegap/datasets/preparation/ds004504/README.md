# ds004504 – dataset preparation

Scripts that create the ImputeGAP dataset `ds004504_sub-001` from OpenNeuro
[ds004504](https://openneuro.org/datasets/ds004504) — *A dataset of EEG recordings from:
Alzheimer's disease, Frontotemporal dementia and Healthy subjects*
(Miltiadous et al., 2023, doi: [10.3390/data8060095](https://doi.org/10.3390/data8060095), license CC0).

| File | Purpose |
|---|---|
| `download.py` | Downloads the selected subjects from OpenNeuro into `data/raw/ds004504/` |
| `prepare.py` | Converts each subject into an ImputeGAP dataset file |
| `ds004504.py` | Loading, cropping, resampling and saving logic used by `prepare.py` |

`data/` sits at the repository root and is excluded from git.


## Usage

Run from the repository root with the project environment (needs `mne`, `pandas` and, for downloading, `openneuro-py`):

```bash
python imputegap/datasets/preparation/ds004504/download.py --subjects sub-001
python imputegap/datasets/preparation/ds004504/prepare.py --subjects sub-001
```

`prepare.py` takes the preprocessed (derivative) eyes-closed recording, keeps the first 40 seconds
and resamples it from 500 Hz to 128 Hz. It writes:

| Output | Location | Content |
|---|---|---|
| `ds004504_sub-001.txt` | `imputegap/datasets/` | 5120 time steps × 19 channels, space-separated, in volts |
| `sub-001.json` | `data/processed/ds004504/` | channel names and 3D positions, sampling rates, group, age, gender, MMSE |

The dataset is registered in `tools/utils.py` (`list_of_datasets`) and documented in
`imputegap/datasets/README.md`, so it loads like any built-in dataset:

```python
ts.load_series(utils.search_path("ds004504_sub-001"), normalizer="z_score")
```
