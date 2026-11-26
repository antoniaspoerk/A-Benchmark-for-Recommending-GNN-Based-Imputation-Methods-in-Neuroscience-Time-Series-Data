import os

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "imputegap_assets"))

from imputegap.recovery.benchmark import BenchGAP
from imputegap.tools import utils

my_algorithms = ["SPIN"]

my_opt = "default_params"

my_datasets = ["ds004504_sub-001","eeg-alcohol"]

my_patterns = ["disjoint"]

range = [0.2]

# launch the evaluation
bench = BenchGAP()
bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, nbr_vals=256,optimizer=my_opt, save_dir=os.path.join(ASSETS, "benchmark"))

# display the plot
bench.subplots.show()