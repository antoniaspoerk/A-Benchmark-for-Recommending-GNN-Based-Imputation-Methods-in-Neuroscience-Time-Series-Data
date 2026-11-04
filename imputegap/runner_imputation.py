import os

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "imputegap_assets"))

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

# initialize the time series object
ts = TimeSeries()
print(f"\nImputation algorithms : {ts.algorithms}")

# load and normalize the dataset
ts.load_series(utils.search_path("ds004504_sub-001"),nbr_series=19, nbr_val=100, normalizer=None)

# contaminate the time series
ts_m = GenGap.disjoint(ts.data, rate_series=0.25, limit=1, offset=0.1, logic_by_series=True, verbose=True)

# impute the contaminated series
#imputer = Imputation.MatrixCompletion.CDRec(ts_m)
imputer = Imputation.Statistics.WindowImpute(ts_m)
imputer.impute()

# compute and print the imputation metrics
imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, subplot=True, algorithm=imputer.algorithm, save_path=os.path.join(ASSETS, "imputation"))