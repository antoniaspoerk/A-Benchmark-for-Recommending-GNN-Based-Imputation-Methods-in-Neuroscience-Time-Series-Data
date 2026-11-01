import os

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "imputegap_assets"))

from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nMissingness patterns : {ts.patterns}")

# load and normalize the dataset
ts.load_series(utils.search_path("ds004504_sub-001"), normalizer="z_score")

# contaminate the time series with blackout pattern
ts_m = GenGap.blackout(ts.data, rate_series=0.4, logic_by_series=True, verbose=True)

# plot the contaminated time series
ts.plot(ts.data, ts_m, nbr_series=19,nbr_val=100, subplot=True, save_path=os.path.join(ASSETS, "contamination"))