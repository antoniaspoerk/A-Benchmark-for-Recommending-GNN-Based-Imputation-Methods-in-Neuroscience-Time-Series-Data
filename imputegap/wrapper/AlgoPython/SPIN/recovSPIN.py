import copy
import datetime
import os

import numpy as np
import torch

import imputegap.tools.utils as utils_imp
from imputegap.wrapper.AlgoPython.SPIN.lib.data.dataset_imputegap import get_dataloader
from imputegap.wrapper.AlgoPython.SPIN.lib.models.spin import SPINModel
from imputegap.wrapper.AlgoPython.SPIN.lib.scheduler import CosineSchedulerWithRestarts

def handle_parser(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description='SPIN')
    parser.add_argument('--whiten-prob', type=float, nargs='+', default=[0.2, 0.5, 0.8])
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--p-noise', type=float, default=0.25) 
    parser.add_argument('--message-layers', type=int, default=1)
    parser.add_argument('--reweight', type=str, default='softmax')

    args, _unknown = parser.parse_known_args(argv)

    return args


def masked_l1(y_hat, y, mask):
    mask = mask.float()
    return (torch.abs(y_hat - y) * mask).sum() / mask.sum().clamp(min=1)


def forward(model, batch, edge_index, edge_weight):
    return model(batch["x"], batch["u"], batch["mask"], edge_index, edge_weight)


def evaluate(model, loader, edge_index, edge_weight, device):
    model.eval()
    err, cnt = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x_hat, _ = forward(model, batch, edge_index, edge_weight)
            m = batch["eval_mask"].float()
            err += (torch.abs(x_hat - batch["y"]) * m).sum().item()
            cnt += m.sum().item()
    return err/cnt if cnt > 0 else float("nan")


def train(model, train_loader, valid_loader, edge_index, edge_weight, device, epochs, lr, patience, batches_epoch, whiten_prob, grad_clip_val, verbose, split_batch_in=2, precision=16):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineSchedulerWithRestarts(optimizer, num_warmup_steps=12, min_factor=0.1, linear_decay=0.67,num_training_steps=epochs, num_cycles=max(1, epochs // 100))
    use_amp = precision == 16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    whiten_prob = torch.tensor(whiten_prob, device=device)
    best_val, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(epochs):
        model.train()
        losses = []
        optimizer.zero_grad() 
        pending = 0 
        for i, batch in enumerate(train_loader):
            if i >= batches_epoch*split_batch_in:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            mask = batch["mask"].bool()
            p = whiten_prob[torch.randint(len(whiten_prob), (mask.size(0),), device=device)].view(-1, 1, 1, 1)
            whiten_mask = torch.rand(mask.shape, device=device) > p
            batch["mask"] = mask & whiten_mask
            injected_missing = mask & ~batch["mask"]
            batch["x"] = batch["y"] * batch["mask"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                x_hat, imputations = forward(model, batch, edge_index, edge_weight)
                loss = masked_l1(x_hat.float(), batch["y"], injected_missing)
                for imp in imputations:
                    loss = loss + masked_l1(imp.float(), batch["y"], injected_missing)
            scaler.scale(loss / split_batch_in).backward()
            pending += 1
            if pending == split_batch_in:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                pending = 0
            losses.append(loss.item())
        if pending > 0:  
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()
        val_mae = evaluate(model, valid_loader, edge_index, edge_weight, device)
        if verbose:
            print(f"\tepoch {epoch + 1}/{epochs}  train_loss {np.mean(losses):.4f}  val_mae {val_mae:.4f}")
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


def reconstruct(model, imputegap_loader, edge_index, edge_weight, scaler, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in imputegap_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            x_hat, _ = forward(model, batch, edge_index, edge_weight)
            preds.append(x_hat[..., 0].cpu())
    preds = torch.cat(preds, 0)  # [windows, steps, nodes]
    return preds * scaler["std"][:, 0] + scaler["mean"][:, 0]  

def prepare_deep_learning_params(incomp_data, seq_len, batch_size, sliding_windows, tr_ratio, verbose):
    # was utils.prepare_deep_learning_params in ImputeGAP 1.1.2 (removed in 1.1.3); same logic
    error = False
    multivariate = sliding_windows == 0
    if seq_len == -1 or batch_size == -1:
        if multivariate:
            seq_len, batch_size = utils_imp.auto_seq_sample(matrix=incomp_data, tr_ratio=tr_ratio, verbose=verbose)
            sliding_windows = seq_len
        else:
            seq_len, batch_size = utils_imp.auto_seq_llms(data_x=incomp_data, goal="seq", subset=True, high_limit=50, verbose=verbose)
    if seq_len > len(incomp_data):
        print(f"(ERROR) The current seq_length {seq_len} is not adapted to the contaminated matrix {len(incomp_data)} !")
        error = True
    return multivariate, seq_len, batch_size, sliding_windows, error


def recovSPIN(incomp_data, seq_len=24, batch_size=8, epochs=30, sliding_windows=1, sim_type="corr", hidden_size=32,
              n_layers=4, eta=3, lr=0.0008, patience=40, batches_epoch=300, num_workers=0, tr_ratio=0.7, seed=42,
              verbose=True, replicat=False, dataset_name="air36", split_batch_in=2, precision=16):
    args = handle_parser()
    recov = np.copy(incomp_data)
    m_mask = np.isnan(incomp_data)
    multivariate, seq_len, batch_size, sliding_windows, error = prepare_deep_learning_params(incomp_data=incomp_data, seq_len=seq_len, batch_size=batch_size, sliding_windows=sliding_windows, tr_ratio=tr_ratio, verbose=verbose)
    if error:
        return incomp_data
    stride = seq_len if multivariate else sliding_windows  
    test_len, val_len = utils_imp.sets_splitter_based_on_training(tr_ratio, verbose=verbose)
    T = incomp_data.shape[0]
    pad = (-(T - seq_len)) % stride
    ts_m = np.vstack([incomp_data, np.full((pad, incomp_data.shape[1]), np.nan)]) if pad else np.copy(incomp_data)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    logdir = os.path.join(here, "imputegap_assets/logs/spin/", f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{seed}")
    os.makedirs(logdir, exist_ok=True)

    if verbose:
        print(f"\n(IMPUTATION) SPIN\n\tMatrix: {incomp_data.shape[0]}, {incomp_data.shape[1]}\n\tseq_len: {seq_len}\n\tbatch_size: {batch_size}\n\tepochs: {epochs}\n\tsliding_windows: {stride}\n\tsim_type: {sim_type}\n\thidden_size: {hidden_size}\n\tn_layers: {n_layers}\n\teta: {eta}\n\tlr: {lr}\n\tpatience: {patience}\n\tbatches_epoch: {batches_epoch}\n\tnum_workers: {num_workers}\n\ttr_ratio: {tr_ratio}\n\tval_ratio: {val_len}\n\tts_ratio: {test_len}\n\tdevice: {device}\n")
        print(f"call: spin.impute(params={{'seq_len': {seq_len}, 'batch_size': {batch_size}, 'epochs': {epochs}, 'sliding_windows': {sliding_windows}, 'sim_type': '{sim_type}', 'hidden_size': {hidden_size}, 'n_layers': {n_layers}, 'eta': {eta}, 'lr': {lr}, 'patience': {patience}, 'batches_epoch': {batches_epoch}, 'num_workers': {num_workers}}})\n")
        print("\npre-processing...")
    train_loader, valid_loader, test_loader, imputegap_loader, edge_index, edge_weight, scaler = get_dataloader(
        ts_m, seq_len=seq_len, batch_size=batch_size // split_batch_in, stride=stride, sim_type=sim_type, tr_ratio=tr_ratio,
        val_len=val_len, test_len=test_len, p_noise=args.p_noise, num_workers=num_workers, verbose=verbose)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
    model = SPINModel(input_size=1, hidden_size=hidden_size, n_nodes=ts_m.shape[1], u_size=4, output_size=1,
                      n_layers=n_layers, eta=eta, message_layers=args.message_layers, temporal_self_attention=True,
                      reweight=args.reweight).to(device)

    if verbose:
        print("\ntraining...")
    model = train(model, train_loader, valid_loader, edge_index, edge_weight, device, epochs=epochs, lr=lr, patience=patience,
                  batches_epoch=batches_epoch, whiten_prob=args.whiten_prob, grad_clip_val=args.grad_clip_val, verbose=verbose,split_batch_in=split_batch_in,precision=precision)
    torch.save(model.state_dict(), os.path.join(logdir, "spin_best.pt"))

    if verbose:
        print(f"\ntesting...\n\ttest_mae (scaled): {evaluate(model, test_loader, edge_index, edge_weight, device):.4f}")
        print("\nreconstruct...")
    preds = reconstruct(model, imputegap_loader, edge_index, edge_weight, scaler, device)
    imputed_matrix = utils_imp.reconstruction_window_based(preds=preds, nbr_timestamps=ts_m.shape[0], sliding_windows=stride, verbose=verbose, deep_verbose=False)
    imputed_matrix = np.asarray(imputed_matrix)[:T]
    recov[m_mask] = imputed_matrix[m_mask]
    return recov