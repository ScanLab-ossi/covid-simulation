import unittest
import pandas as pd
import pandas.testing as pdt

from simulation.sensitivity_analysis import Analysis
from simulation.dataset import Dataset
from simulation.output import Batch
from simulation.task import Task
from simulation.constants import *


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("copenhagen_hops")
        task = Task(test=True)
        self.batch = Batch(dataset, task)
        self.batch.load(TEST_FOLDER / "mock_summed_batch.csv", format_="csv")
        self.analysis = Analysis(dataset, task)

    @unittest.skipIf(True, "Dunno what happened")  # FIXME: fix count test
    def test_count(self):
        pdt.assert_frame_equal(
            self.analysis.count(
                self.batch, grouping="sick", percent=20, how="day", avg=False
            ),
            pd.DataFrame(
                {"value": [11, 11, 11], "metric": ["day_of_specific_percent_sick"] * 3}
            ),
        )
        self.assertAlmostEqual(
            self.analysis.count(
                self.batch, grouping="infectors", amount=41, how="day", avg=True
            ),
            8.6,
            delta=0.1,
        )
        self.assertAlmostEqual(
            self.analysis.count(
                self.batch, grouping="infected", max_=True, how="amount", avg=True
            ),
            133,
            delta=0.1,
        )

    def test_r_0(self):
        pass
        # TODO: build test for r_0
