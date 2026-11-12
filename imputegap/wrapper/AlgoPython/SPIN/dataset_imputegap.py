import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 
from torch_geometric.utils import dense_to_sparse

def time_encoding(n_steps, step_seconds=300): #5 minutes per time step
    seconds = np.arange(n_steps)*step_seconds
    u = []
    for period in (24*60*60, 7*24*60*60):
        angle = 2*np.pi*seconds/period
        u += [np.sin(angle), np.cos(angle)]
    return np.stack(u, axis=-1).astype(np.float32)

def build_edge_index(data, mask, sim_type="corr", thr=0.1):
    n = data.shape[1]
    if sim_type == "uniform":
        adj = np.ones((n, n), dtype=np.float32)
    elif sim_type == "corr":
        x = np.where(mask, data, np.nan)
        x = np.where(np.isnan(x), np.nanmean(x, axis=0, keepdims=True), x)  
        adj = np.nan_to_num(np.corrcoef(x.T), nan=0.0)
        adj = np.maximum(adj, 0.0) 
        adj[adj < thr] = 0.0
    else:
        raise ValueError(f"sim_type '{sim_type}' not available, use 'corr' or 'uniform'.")
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)
    edge_index, _ = dense_to_sparse(torch.as_tensor(adj))
    return edge_index


class ImputeGAPDataset(Dataset):
    def __init__(self, data, mask, eval_mask, u, starts, seq_len):
        self.data = torch.as_tensor(data, dtype=torch.float32)[..., None]
        self.mask = torch.as_tensor(mask)[..., None]
        self.eval_mask = torch.as_tensor(eval_mask)[..., None]
        self.u = torch.as_tensor(u)
        self.starts = list(starts)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = slice(self.starts[i], self.starts[i] + self.seq_len)
        mask = self.mask[s]
        return dict(x=self.data[s] * mask, y=self.data[s], mask=mask, eval_mask=self.eval_mask[s], u=self.u[s])


def window_starts(first, last, seq_len, stride):
    return range(first, max(first, last - seq_len) + 1, stride)


def get_dataloader(ts_m, seq_len, batch_size, stride=1, sim_type="corr", tr_ratio=0.7, val_len=0.1,test_len=0.2, p_noise=0.25, num_workers=0, seed=56789, verbose=True):
    T, N = ts_m.shape
    rng = np.random.default_rng(seed)
    ob_mask = ~np.isnan(ts_m)
    eval_mask = (rng.random(ts_m.shape) < p_noise) & ob_mask
    training_mask = ob_mask & ~eval_mask
    train_end = int(round(T * tr_ratio))
    val_end = int(round(T * (tr_ratio + val_len)))
    train_vals = np.where(training_mask[:train_end], ts_m[:train_end], np.nan)
    mean = np.nanmean(train_vals, axis=0)
    std = np.nanstd(train_vals, axis=0)
    std[std == 0] = 1.0
    data = np.nan_to_num((ts_m - mean) / std, nan=0.0)

    u = time_encoding(T)
    edge_index = build_edge_index(ts_m[:train_end], training_mask[:train_end], sim_type=sim_type)

    def make(starts, mask, shuffle=False, drop_last=False):
        ds = ImputeGAPDataset(data, mask, eval_mask, u, starts, seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    train_loader = make(window_starts(0, train_end, seq_len, 1), training_mask, shuffle=True, drop_last=True)
    valid_loader = make(window_starts(train_end, val_end, seq_len, seq_len), training_mask)
    test_loader = make(window_starts(val_end, T, seq_len, seq_len), training_mask)
    imputegap_loader = make(range(0, T - seq_len + 1, stride), ob_mask)

    if verbose:
        print(f"\tdataset: {T} time steps, {N} nodes, {edge_index.shape[1]} edges ({sim_type})")
        print(f"\t(TRAIN) {len(train_loader.dataset)} | (VAL) {len(valid_loader.dataset)} | (TEST) {len(test_loader.dataset)} | (RECONSTRUCTION) {len(imputegap_loader.dataset)} windows of size {seq_len}")

    scaler = dict(mean=torch.as_tensor(mean, dtype=torch.float32)[:, None], std=torch.as_tensor(std, dtype=torch.float32)[:, None])
    return train_loader, valid_loader, test_loader, imputegap_loader, edge_index, scaler