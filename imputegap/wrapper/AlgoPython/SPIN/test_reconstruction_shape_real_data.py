import warnings; warnings.filterwarnings("ignore")
import numpy as np
from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap

T, N = 300, 10
data = np.arange(1, T * N + 1, dtype=float).reshape(T, N)    # row 1: 1-10, row 2: 11-20, row 3: 21-30
incomp = GenGap.mcar(data)  #contaminate data

alg = Imputation.DeepLearning.SPIN(incomp)
alg.impute(params={"seq_len": 24, "batch_size": 8, "epochs": 1, "sliding_windows": 1, "sim_type": "corr",
                    "hidden_size": 32, "n_layers": 4, "eta": 3, "lr": 0.0008, "patience": 40,
                    "batches_epoch": 2, "num_workers": 0})
recov = alg.recov_data

missing = np.isnan(incomp)
print("shape:", recov.shape, " NaN left:", np.isnan(recov).sum(), " 999 left:", (recov == 999).sum())
print("observed values unchanged:", np.array_equal(recov[~missing], incomp[~missing]))
r, c = np.argwhere(missing)[0]
print(f"first gap at row {r}, column {c}: true {data[r, c]:.0f}, imputed {recov[r, c]:.1f}")
print("mean absolute error on gaps:", np.abs(recov[missing] - data[missing]).mean())
print("999 in the gaps:", (recov[missing] == 999).sum())