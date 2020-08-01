import unittest
import pandas as pd
import pandas.testing as pdt

from simulation.analysis import Analysis
from simulation.dataset import Dataset
from simulation.output import Output
from simulation.task import Task
from simulation.constants import *


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        task = Task()
        self.output = Output(dataset, task)
        output_df = pd.read_pickle(TEST_FOLDER / "mock_output_df.pkl",)
        self.output.batch = [output_df] * 3
        self.analysis = Analysis(dataset, task, self.output)

    def sick(self):
        for what in ["max_amount", "max_day"]:
            res = self.analysis.sick(self.output, what=what)
            pdt.assert_frame_equal(
                res, pd.DataFrame({"sick": [(0 if what == "max_day" else 10.0)] * 3}),
            )

    def test_infected(self):
        for what in ["max_amount", "max_day"]:
            res = self.analysis.infected(self.output, what=what)
            pdt.assert_frame_equal(
                res, pd.DataFrame({"infected": [(0 if what == "max_day" else 10)] * 3}),
            )

    def test_r_0(self):
        pass
        # TODO
