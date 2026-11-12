import time 

from imputegap.wrapper.AlgoPython.SPIN.recovSPIN import recovSPIN

def spin(incomp_data, seq_len=24, batch_size=8, epochs=30, sliding_windows=1, sim_type="corr", hidden_size=32, n_layers=4, eta=3, lr=0.0008, patience=40, batches_epoch=300, num_workers=0, tr_ratio=0.7, seed=42, logs=True, verbose=True):
    start_time=time.time()

    recov_data = recovSPIN(incomp_data, seq_len=seq_len, batch_size=batch_size, epochs=epochs, sliding_windows=sliding_windows, sim_type=sim_type, hidden_size=hidden_size, n_layers=n_layers, eta=eta, lr=lr, patience=patience, batches_epoch=batches_epoch, num_workers=num_workers, tr_ratio=tr_ratio, seed=seed, verbose=verbose)

    end_time=time.time()
    if logs and verbose: 
        print(f"\n> logs: imputation spin - Execution Time: {(end_time - start_time):.4f} seconds\n")
    return recov_data
