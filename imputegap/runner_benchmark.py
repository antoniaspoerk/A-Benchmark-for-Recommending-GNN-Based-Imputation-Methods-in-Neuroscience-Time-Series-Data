import os

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "imputegap_assets"))

from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

my_algorithms = ["SPIN","GRIN","TimesNet"]

my_opt = "default_params"

my_datasets = ["chlorine"]

my_patterns = ["aligned"]

range = [0.2]

# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, optimizer=my_opt, save_dir=os.path.join(ASSETS, "benchmark"))

# display the plot
bench.subplots.show()