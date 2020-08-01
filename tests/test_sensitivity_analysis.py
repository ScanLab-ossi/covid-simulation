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


# self.assertEqual(res.columns, )
