import argparse
import os
import sys

import numpy as np
import torch

SPIN_REPO = os.path.expanduser("~/spin-original/spin")
sys.path.insert(0, SPIN_REPO)
sys.path.insert(0, os.path.join(SPIN_REPO, "experiments"))
os.chdir(SPIN_REPO)  # tsl reads tsl_config.yaml from the working directory

from tsl.data import ImputationDataset, SpatioTemporalDataModule  # noqa: E402
from tsl.data.preprocessing import StandardScaler  # noqa: E402

from run_imputation import get_dataset  # noqa: E402  (authors' function, unchanged)

# values from the authors' config/imputation/spin.yaml and run_imputation.py defaults
WINDOW, STRIDE = 24, 1
VAL_LEN, TEST_LEN = 0.1, 0.2
ADJ_THRESHOLD = 0.1
ALL = ["air36", "la_point", "la_block", "bay_point", "bay_block"]


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_channel(a):
    a = to_np(a)
    return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a


def export(name, out_dir):
    dataset = get_dataset(name)

    # same as run_imputation.py, SPIN branch
    time_emb = dataset.datetime_encoded(["day", "week"]).values
    adj = dataset.get_connectivity(threshold=ADJ_THRESHOLD, include_self=False, force_symmetric=True)
    data, idx = dataset.numpy(return_idx=True)

    torch_dataset = ImputationDataset(data, idx,
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=adj,
                                      exogenous={"global_temporal_encoding": time_emb},
                                      input_map={"u": "temporal_encoding", "x": "data"},
                                      window=WINDOW, stride=STRIDE)
    splitter = dataset.get_splitter(val_len=VAL_LEN, test_len=TEST_LEN)
    dm = SpatioTemporalDataModule(torch_dataset, scalers={"data": StandardScaler(axis=(0, 1))},
                                  splitter=splitter, batch_size=4)
    dm.setup()

    # graph: tsl returns (edge_index, edge_weight) or a dense matrix depending on version
    if isinstance(adj, (tuple, list)):
        edge_index, edge_weight = to_np(adj[0]), to_np(adj[1])
    else:
        dense = to_np(adj)
        edge_index = np.stack(np.nonzero(dense)).astype(np.int64)
        edge_weight = dense[edge_index[0], edge_index[1]]

    window_starts = getattr(torch_dataset, "indices", None)
    if window_starts is None:
        window_starts = torch_dataset._indices
    scaler = dm.scalers["data"]

    def split_idx(name):  # window indices of one split (attribute names differ across tsl versions)
        idx = getattr(splitter, f"{name}_idxs", None)
        if idx is None:
            idx = getattr(dm, f"{name}set").indices
        return idx

    out = dict(
        data=squeeze_channel(data).astype(np.float32),                         # (T, N), raw values
        training_mask=squeeze_channel(dataset.training_mask).astype(bool),     # observed & not held out
        eval_mask=squeeze_channel(dataset.eval_mask).astype(bool),             # held-out points scored in the paper
        u=np.asarray(time_emb, dtype=np.float32),                              # (T, 4) day/week encoding
        edge_index=edge_index.astype(np.int64),
        edge_weight=np.asarray(edge_weight, dtype=np.float32),
        window_starts=to_np(window_starts).astype(np.int64),                   # start step of every window
        train_idx=to_np(split_idx("train")).astype(np.int64),                  # window indices per split
        val_idx=to_np(split_idx("val")).astype(np.int64),
        test_idx=to_np(split_idx("test")).astype(np.int64),
        scaler_bias=to_np(scaler.bias).astype(np.float32).reshape(-1),
        scaler_scale=to_np(scaler.scale).astype(np.float32).reshape(-1),
        window=np.int64(WINDOW),
    )
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(path, **out)

    T, N = out["data"].shape
    print(f"{name}: T={T} N={N} edges={out['edge_index'].shape[1]} "
          f"windows train/val/test={len(out['train_idx'])}/{len(out['val_idx'])}/{len(out['test_idx'])} "
          f"eval points={int(out['eval_mask'].sum())} scaler mean={out['scaler_bias']} std={out['scaler_scale']}")
    print(f"  -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", help=f"one of {ALL} or 'all'")
    p.add_argument("--out-dir", default=os.path.expanduser("~/spin-original/validation/exported"))
    a = p.parse_args()
    for n in (ALL if a.dataset == "all" else [a.dataset]):
        export(n, a.out_dir)
