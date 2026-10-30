#OpenNeuro ds004504 (Alzheimer's/FTD/control) -> format for ImputeGAP: matrix X of shape [T, N] (time steps x channels)

import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

GROUP_LABELS = {"A": "alzheimer", "F": "ftd", "C": "control"}


def find_eeg_file(raw_dir: Path, subject: str, use_derivatives: bool = True) -> Path:
    """Return the .set file of a subject (preprocessed derivative by default)."""
    base = raw_dir / "derivatives" / subject if use_derivatives else raw_dir / subject
    files = sorted(base.rglob("*_eeg.set"))
    if not files:
        raise FileNotFoundError(f"No *_eeg.set found under {base}")
    return files[0]


def load_labels(raw_dir: Path) -> dict:
    """Map participant_id -> {group, age, gender, mmse} from participants.tsv."""
    df = pd.read_csv(raw_dir / "participants.tsv", sep="\t")
    labels = {}
    for _, row in df.iterrows():
        labels[row["participant_id"]] = {
            "group": GROUP_LABELS.get(row.get("Group"), row.get("Group")),
            "age": row.get("Age"),
            "gender": row.get("Gender"),
            "mmse": row.get("MMSE"),
        }
    return labels


def load_subject(raw_dir: Path, subject: str, sfreq: float | None = 128.0,
                 tmin: float = 0.0, duration: float | None = 40.0,
                 use_derivatives: bool = True):
    """Load one subject as X [T, N] plus a metadata dict."""
    path = find_eeg_file(raw_dir, subject, use_derivatives)
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
    raw.pick("eeg")
    orig_sfreq = raw.info["sfreq"]

    # crop first (cheaper), then resample
    if duration is not None:
        tmax = min(tmin + duration, raw.times[-1])
        raw.crop(tmin=tmin, tmax=tmax)
    if sfreq is not None and sfreq != orig_sfreq:
        raw.resample(sfreq, verbose="error")

    X = raw.get_data().T  # MNE: (channels, time) -> ImputeGAP: (time, channels)
    if duration is not None:
        X = X[: int(round(duration * raw.info["sfreq"]))]

    if np.isnan(X).any():
        raise ValueError(f"{subject}: recording already contains NaNs")

    montage = raw.get_montage()
    positions = montage.get_positions()["ch_pos"] if montage is not None else {}

    labels = load_labels(raw_dir).get(subject, {})
    meta = {
        "dataset": "ds004504",
        "subject": subject,
        "source_file": str(path),
        "ch_names": raw.ch_names,
        "ch_pos": {ch: list(map(float, p)) for ch, p in positions.items()},
        "orig_sfreq": orig_sfreq,
        "sfreq": raw.info["sfreq"],
        "n_times": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "duration_s": X.shape[0] / raw.info["sfreq"],
        "unit": "V",
        **labels,
    }
    return X, meta


def save_subject(X: np.ndarray, meta: dict, out_dir: Path, txt_dir: Path) -> Path:
    """Write ImputeGAP txt (space-delimited, no header) to txt_dir + metadata json to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = txt_dir / f"{meta['dataset']}_{meta['subject']}.txt"  # e.g. ds004504_sub-001.txt
    np.savetxt(txt_path, X, delimiter=" ", fmt="%.9e")  # %.9e: values are ~1e-5 V
    with open(out_dir / f"{meta['subject']}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return txt_path