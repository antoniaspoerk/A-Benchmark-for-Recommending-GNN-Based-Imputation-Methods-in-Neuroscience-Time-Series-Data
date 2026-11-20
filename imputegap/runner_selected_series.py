import os
import matplotlib.pyplot as plt
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.tools import utils

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "imputegap_assets", "recovery_plots"))
os.makedirs(ASSETS, exist_ok=True)

my_algorithms = ["SPIN", "GRIN"]

#what series what timepoints
selected = {
    "outliers_vs_typical": {"series": [1, 2, 5], "timepoints": (0, 300)},
    "easy_section_all_series": {"series": list(range(1, 20)), "timepoints": (3200, 3500)},
    "hard_section_all_series": {"series": list(range(1, 20)), "timepoints": (3900, 4200)},
    "long_view": {"series": [1, 5, 9, 17], "timepoints": (0, 2000)},
    "without_outliers": {"series": [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19], "timepoints": (0, 300)},
    "series 1-5": {"series": [1, 2, 3, 4, 5], "timepoints": (0, 300)},
}

full = TimeSeries()
full.load_series(utils.search_path("ds004504_sub-001"), normalizer=None, verbose=False)

for area in selected:
    series = selected[area]["series"]
    first, last = selected[area]["timepoints"]
    series_text = "all 19 series" if len(series) == 19 else "series " + ", ".join(str(s) for s in series)

    # build this area of the dataset, normalize it and contaminate it
    ts = TimeSeries()
    ts.import_matrix(full.data[first:last, [s - 1 for s in series]])
    ts.normalize(normalizer="z_score", verbose=False)
    incomp_data = GenGap.disjoint(ts.data, verbose=False)

    for algorithm in my_algorithms:
        imputer = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=algorithm, verbose=False)
        imputer.logs = False
        imputer.impute()
        imputer.score(ts.data)

        ts.plot(input_data=ts.data, incomp_data=incomp_data, recov_data=imputer.recov_data, subplot=True, algorithm=imputer.algorithm, cont_rate=area, save_path=None, display=False, verbose=False)
        fig = plt.gcf()
        fig.suptitle(f"{algorithm} on {area}: {series_text}, time points {first} to {last} (RMSE = {imputer.metrics['RMSE']:.3f}, MAE = {imputer.metrics['MAE']:.3f})", fontsize=12, y=1.03)
        fig.savefig(os.path.join(ASSETS, f"{area}_{imputer.algorithm}_plot.jpg"), bbox_inches="tight")
        plt.close(fig)
        print(f"{area}, {algorithm}: RMSE = {imputer.metrics['RMSE']:.3f}, MAE = {imputer.metrics['MAE']:.3f}")

print("\nplots saved in:", ASSETS)