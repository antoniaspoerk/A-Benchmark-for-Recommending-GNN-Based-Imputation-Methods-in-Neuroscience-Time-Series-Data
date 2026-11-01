import os

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "imputegap_assets"))

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.contamination import GenGap
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.downstream import Downstream
from imputegap.tools import utils

_plot_downstream = Downstream._plot_downstream
def _plot_to_assets(*args, **kwargs):
    kwargs["save_path"] = os.path.join(ASSETS, "downstream")
    return _plot_downstream(*args, **kwargs)
Downstream._plot_downstream = staticmethod(_plot_to_assets)

# initialize the time series object
ts = TimeSeries()
print(f"\nImputeGAP downstream models for forcasting : {ts.forecasting_models}")

# load and normalize the dataset
ts.load_series(utils.search_path("ds004504_sub-001"), normalizer="z_score")

# contaminate the time series
ts_m = GenGap.aligned(ts.data, rate_series=0.7) # type: ignore

# define and impute the contaminated series
imputer = Imputation.DeepLearning.BitGraph(ts_m)
imputer.impute()

# compute and print the downstream results
downstream_config = {"task": "forecast", "model": "prophet", "baseline": "ZeroImpute", "split": 0.8}
imputer.score(ts.data, imputer.recov_data, downstream=downstream_config)
ts.print_results(imputer.downstream_metrics, text="Downstream results")