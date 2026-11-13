import numpy as np
import torch
from imputegap.tools import utils as utils_imp
from imputegap.wrapper.AlgoPython.SPIN.dataset_imputegap import get_dataloader

T, N, seq_len = 300, 10, 24 #t=300, n=10, window = 24
data = np.arange(1, T * N + 1, dtype=float).reshape(T, N)   # row 1: 1-10, row 2: 11-20, row 3: 21-30 ...
ts_m = data.copy()
ts_m[50:60, 2] = np.nan                                   

_, _, _, imp, _, scaler = get_dataloader(ts_m, seq_len=seq_len, batch_size=8, stride=1, verbose=False) #train_loader, valid_loader, test_loader, imputegap_loader, edge_index, scaler

windows = torch.cat([b["y"][..., 0] for b in imp], 0)  #true windows
windows = windows * scaler["std"][:, 0] + scaler["mean"][:, 0]    # scale back to original values
rec = np.asarray(utils_imp.reconstruction_window_based(preds=windows, nbr_timestamps=T, sliding_windows=1, verbose=False)) #assemble windows back to matrix

observed = ~np.isnan(ts_m)
print("shape:", rec.shape, "max deviation:", np.abs(rec - data)[observed].max()," observed values identical:", np.allclose(rec[observed], data[observed], atol=1e-3))