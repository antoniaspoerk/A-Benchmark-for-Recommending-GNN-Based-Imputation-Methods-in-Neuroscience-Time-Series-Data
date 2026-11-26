import os
import unittest
from imputegap.tools import utils
from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGAP


class TestSPIN(unittest.TestCase):

    def test_imputation_spin(self, name="spin", limit=0.10):
        """
        the goal is to test if only the simple imputation with the technique has the expected outcome
        """
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "toml/imputegap_results.toml")

        dataset, rmse, mae = utils.get_resuts_unit_tests(algo_name=name, loader=path)

        ts = TimeSeries()
        ts.load_series(utils.search_path(dataset), nbr_series=19, nbr_val=500, normalizer="z_score")

        incomp_data = GenGAP.mcar(ts.data)
        algo = utils.config_impute_algorithm(incomp_data=incomp_data, algorithm=name, verbose=True)
        algo.impute(params={"seq_len": 24, "batch_size": 8, "epochs": 2, "sliding_windows": 1, "sim_type": "corr",
                            "hidden_size": 32, "n_layers": 4, "eta": 3, "lr": 0.0008, "patience": 40,
                            "batches_epoch": 5, "num_workers": 0})
        algo.score(ts.data)
        metrics = algo.metrics

        print(f"{name}:{metrics = }\n")


        expected_metrics = {"RMSE": rmse, "MAE": mae}

        self.assertTrue(abs(metrics["RMSE"] - expected_metrics["RMSE"]) < limit, f"metrics RMSE = {metrics['RMSE']}, expected RMSE = {expected_metrics['RMSE']} ")
        self.assertTrue(abs(metrics["MAE"] - expected_metrics["MAE"]) < limit, f"metrics MAE = {metrics['MAE']}, expected MAE = {expected_metrics['MAE']} ")


if __name__ == "__main__":
            unittest.main()