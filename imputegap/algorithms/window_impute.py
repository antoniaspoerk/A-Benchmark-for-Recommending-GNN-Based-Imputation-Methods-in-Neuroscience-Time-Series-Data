import numpy as np


def window_impute(incomp_data, window=10, stat="mean"):

    # core of the algorithm
    mean_value = np.nanmean(incomp_data) #takes everything except NaN values

    # Imputation
    recov_data = incomp_data.copy()
    n_time, n_series = incomp_data.shape
    agg= np.nanmedian if stat == "median" else np.nanmean

    for s in range(n_series): 
        col = recov_data[:,s]
        for t in np.where(np.isnan(incomp_data[:,s]))[0]: 
            nb = col[max(0, t - window):t+window+1]
            recov_data[t, s] = agg(nb) if np.any(~np.isnan(nb)) else np.nanmean(col) #use median or mean to aggregate

    return recov_data


