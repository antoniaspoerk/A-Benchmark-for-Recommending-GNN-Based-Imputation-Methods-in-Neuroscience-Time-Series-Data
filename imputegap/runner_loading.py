import os

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "imputegap_assets"))

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP datasets : {ts.datasets}")

# load and normalize the dataset
ts.load_series(utils.search_path("ds004504_sub-001"), normalizer=None)

# plot and print a subset of time series
ts.plot(input_data=ts.data, nbr_val=100, save_path=ASSETS)