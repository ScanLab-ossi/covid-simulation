import unittest
import pandas as pd
import pandas.testing as pdt
from datetime import date

from simulation.output import Output
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.sensitivity_analysis import (
    SensitivityOutput,
    SensitivityRunner,
    Analysis,
)
from simulation.constants import TEST_FOLDER


class TestSensitivityOutput(unittest.TestCase):
    def test_concat_outputs(self):
        output = SensitivityOutput(Dataset("mock_data"), Task())
        sample_results_df = pd.DataFrame(
            {"peak_infected": [1, 2, 3], "step": [-1] * 3, "parameter": ["D_min"] * 3,}
        )
        output.results = [sample_results_df] * 10
        output.concat_outputs()
        pdt.assert_frame_equal(output.concated, pd.concat([sample_results_df] * 10))


class TestSensitivityRunner(unittest.TestCase):
    pass


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        self.output = Output(dataset, Task())
        output_df = pd.read_pickle(TEST_FOLDER / "mock_output_df.pkl",)
        self.output.batch = [output_df] * 3

    def test_peak_sick(self):
        for what in ["amount", "day"]:
            res = Analysis.peak_sick(self.output, what=what)
            pdt.assert_frame_equal(
                res, pd.DataFrame({"peak_sick": [(0 if what == "day" else 10.0)] * 3}),
            )

    def test_peak_newly_infected(self):
        for what in ["amount", "day"]:
            res = Analysis.peak_newly_infected(self.output, what=what)
            pdt.assert_frame_equal(
                res,
                pd.DataFrame(
                    {"peak_newly_infected": [(0 if what == "day" else 10)] * 3}
                ),
            )


# self.assertEqual(res.columns, )
