import numpy as np
import torch
from imputegap.tools import utils as utils_imp
from imputegap.wrapper.AlgoPython.SPIN.lib.data.dataset_imputegap import get_dataloader

T, N, seq_len = 300, 10, 24 #t=300, n=10, window = 24
data = np.arange(1, T * N + 1, dtype=float).reshape(T, N)   # row 1: 1-10, row 2: 11-20, row 3: 21-30 ...
ts_m = data.copy()
ts_m[50:60, 2] = np.nan                                   

train, val, test, imp, edge_index, edge_weight, scaler = get_dataloader(ts_m, seq_len=24, batch_size=8, stride=1, verbose=False) #train_loader, valid_loader, test_loader, imputegap_loader, edge_index, scaler
print("train window:", next(iter(train))["x"].shape)   # expected [8, 24, 10, 1]
print("imputegap window:", next(iter(imp))["x"].shape) # expected [8, 24, 10, 1]