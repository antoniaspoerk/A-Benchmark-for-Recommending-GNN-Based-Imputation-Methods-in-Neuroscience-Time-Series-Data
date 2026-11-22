#!/usr/bin/env python
#SBATCH --job-name=spin_mine
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-14%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antonia.spoerk@unifr.ch

import argparse
import copy
import csv
import os
import random
import time

import numpy as np
import toml
import torch
from torch.utils.data import DataLoader, Dataset

import imputegap
from imputegap.wrapper.AlgoPython.SPIN import recovSPIN as mine
from imputegap.wrapper.AlgoPython.SPIN.lib.models.spin import SPINModel
from imputegap.wrapper.AlgoPython.SPIN.lib.scheduler import CosineSchedulerWithRestarts

DATASETS = ["air36", "la_point", "la_block", "bay_point", "bay_block"]
SEEDS = [1, 2, 3]


class ExportedWindows(Dataset):
    """Windows over the exported arrays, returning the same dict recovSPIN's loader returns."""

    def __init__(self, scaled, training_mask, eval_mask, u, starts, window):
        self.data = torch.as_tensor(scaled, dtype=torch.float32)[..., None]
        self.mask = torch.as_tensor(training_mask)[..., None]
        self.eval_mask = torch.as_tensor(eval_mask)[..., None]
        self.u = torch.as_tensor(u, dtype=torch.float32)
        self.starts = [int(s) for s in starts]
        self.window = int(window)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = slice(self.starts[i], self.starts[i] + self.window)
        mask = self.mask[s]
        return dict(x=self.data[s] * mask, y=self.data[s], mask=mask, eval_mask=self.eval_mask[s], u=self.u[s])


def train_matched(model, train_loader, valid_loader, edge_index, edge_weight, device, epochs, lr, patience,
                  batches_epoch, whiten_prob, grad_clip_val, accumulate=2, verbose=True):
    """recovSPIN.train with the authors' gradient accumulation + fp16 mixed precision added.
    Lines differing from recovSPIN.train are marked  # <- matched."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineSchedulerWithRestarts(optimizer, num_warmup_steps=12, min_factor=0.1, linear_decay=0.67,
                                            num_training_steps=epochs, num_cycles=max(1, epochs // 100))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")           # <- matched
    whiten_prob = torch.tensor(whiten_prob, device=device)
    best_val, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(epochs):
        model.train()
        losses = []
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            if i >= batches_epoch * accumulate:                                     # <- matched
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            mask = batch["mask"].bool()
            p = whiten_prob[torch.randint(len(whiten_prob), (mask.size(0),), device=device)].view(-1, 1, 1, 1)
            whiten_mask = torch.rand(mask.shape, device=device) > p
            batch["mask"] = mask & whiten_mask
            injected_missing = mask & ~batch["mask"]
            batch["x"] = batch["y"] * batch["mask"]
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):                     # <- matched
                x_hat, imputations = mine.forward(model, batch, edge_index, edge_weight)
                loss = mine.masked_l1(x_hat.float(), batch["y"], injected_missing)
                for imp in imputations:
                    loss = loss + mine.masked_l1(imp.float(), batch["y"], injected_missing)
            scaler.scale(loss / accumulate).backward()                              # <- matched
            if (i + 1) % accumulate == 0:                                           # <- matched
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            losses.append(loss.item())
        scheduler.step()
        val_mae = mine.evaluate(model, valid_loader, edge_index, edge_weight, device)
        if verbose:
            print(f"\tepoch {epoch + 1}/{epochs}  train_loss {np.mean(losses):.4f}  val_mae {val_mae:.4f}", flush=True)
        if np.isnan(val_mae):
            best_state = copy.deepcopy(model.state_dict())
        elif val_mae < best_val:
            best_val, best_state, wait = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"\tearly stopping after epoch {epoch + 1}")
                break
    model.load_state_dict(best_state)
    return model


def test_mae_like_authors(model, loader, edge_index, edge_weight, bias, scale, device):
    """Authors' metric: masked MAE in original units over all test windows (eval_mask points)."""
    model.eval()
    err, cnt = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x_hat, _ = mine.forward(model, batch, edge_index, edge_weight)
            y_hat = x_hat.float() * scale + bias
            y = batch["y"] * scale + bias
            m = batch["eval_mask"].float()
            err += (torch.abs(y_hat - y) * m).sum().item()
            cnt += m.sum().item()
    return err / cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", help="default: chosen from SLURM_ARRAY_TASK_ID")
    p.add_argument("--seed", type=int, help="default: chosen from SLURM_ARRAY_TASK_ID")
    p.add_argument("--mode", choices=["yours", "old", "matched"], default="yours")
    p.add_argument("--exported", default=os.path.expanduser("~/spin-original/validation/exported"))
    p.add_argument("--results", default=os.path.expanduser("~/spin-original/validation/results"))
    p.add_argument("--epochs", type=int, default=None, help="override for a quick smoke test")
    a = p.parse_args()
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if a.dataset is None:
        a.dataset = DATASETS[task // len(SEEDS)]
    if a.seed is None:
        a.seed = SEEDS[task % len(SEEDS)]

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hp = toml.load(os.path.join(os.path.dirname(imputegap.__file__), "env", "default_values.toml"))["spin"]
    if a.epochs is not None:
        hp["epochs"] = a.epochs
    extra = mine.handle_parser([])  # whiten_prob, grad_clip_val, message_layers, reweight (ImputeGAP version defaults)
    print(f"dataset={a.dataset} seed={a.seed} mode={a.mode} device={device} "
          f"gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'}")
    print(f"ImputeGAP version [spin] params: {hp}")
    print(f"ImputeGAP version extra params:whiten_prob={extra.whiten_prob} grad_clip_val={extra.grad_clip_val} "
          f"message_layers={extra.message_layers} reweight={extra.reweight}")

    d = np.load(os.path.join(a.exported, f"{a.dataset}.npz"))
    bias, scale = d["scaler_bias"], d["scaler_scale"]
    scaled = np.nan_to_num((d["data"] - bias) / scale, nan=0.0)
    window = int(d["window"])
    if hp["seq_len"] != window:
        print(f"NOTE: ImputeGAP version seq_len={hp['seq_len']} but the authors' window={window}; using the authors' window.")
    starts = d["window_starts"]

    def loader(split, batch_size, shuffle, drop_last):
        ds = ExportedWindows(scaled, d["training_mask"], d["eval_mask"], d["u"], starts[d[split]], window)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)

    bs = hp["batch_size"]
    train_bs = bs if a.mode == "old" else bs // 2
    train_loader = loader("train_idx", train_bs, True, True)
    valid_loader = loader("val_idx", bs, False, False)
    test_loader = loader("test_idx", bs, False, False)
    edge_index = torch.as_tensor(d["edge_index"]).to(device)
    edge_weight = torch.as_tensor(d["edge_weight"]).to(device)
    N = d["data"].shape[1]
    print(f"T={d['data'].shape[0]} N={N} edges={edge_index.shape[1]} "
          f"train/val/test windows={len(train_loader.dataset)}/{len(valid_loader.dataset)}/{len(test_loader.dataset)}")

    model = SPINModel(input_size=1, hidden_size=hp["hidden_size"], n_nodes=N, u_size=d["u"].shape[1], output_size=1,
                      n_layers=hp["n_layers"], eta=hp["eta"], message_layers=extra.message_layers,
                      temporal_self_attention=True, reweight=extra.reweight).to(device)

    t0 = time.time()
    common = dict(epochs=hp["epochs"], lr=hp["lr"], patience=hp["patience"], batches_epoch=hp["batches_epoch"],
                  whiten_prob=extra.whiten_prob, grad_clip_val=extra.grad_clip_val, verbose=True)
    if a.mode == "yours":    # same arguments recovSPIN() passes since da244850
        model = mine.train(model, train_loader, valid_loader, edge_index, edge_weight, device, **common,
                           split_batch_in=2, precision=16)
    elif a.mode == "old":    # recovSPIN before da244850
        model = mine.train(model, train_loader, valid_loader, edge_index, edge_weight, device, **common,
                           split_batch_in=1, precision=32)
    else:
        model = train_matched(model, train_loader, valid_loader, edge_index, edge_weight, device, **common)
    seconds = time.time() - t0

    t_bias = torch.as_tensor(bias, device=device).view(1, 1, -1, 1) if bias.size == N else float(bias.reshape(-1)[0])
    t_scale = torch.as_tensor(scale, device=device).view(1, 1, -1, 1) if scale.size == N else float(scale.reshape(-1)[0])
    mae = test_mae_like_authors(model, test_loader, edge_index, edge_weight, t_bias, t_scale, device)
    print(f"Test MAE: {mae:.4f}")

    os.makedirs(a.results, exist_ok=True)
    out = os.path.join(a.results, f"mine_{a.mode}.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "seed", "mode", "test_mae", "seconds", "node"])
        w.writerow([a.dataset, a.seed, a.mode, f"{mae:.4f}", int(seconds), os.uname().nodename])


if __name__ == "__main__":
    main()
