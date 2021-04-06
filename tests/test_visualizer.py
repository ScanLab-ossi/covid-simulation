import unittest
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt
import altair as alt

from simulation.output import Batch, MultiBatch
from simulation.dataset import Dataset
from simulation.visualizer import Visualizer
from simulation.task import Task
from simulation.analysis import Analysis
from simulation.constants import *


class TestVisualizer(unittest.TestCase):
    @patch.object(pd.DataFrame, "to_csv")
    def setUp(self, mock_to_csv):
        self.dataset = Dataset("copenhagen_hops")
        self.task = Task()
        self.batch = Batch(self.dataset, self.task)
        self.batch.load(TEST_FOLDER / "mock_summed_batch.csv", format_="csv")
        self.multibatch = MultiBatch(self.dataset, self.task)
        self.multibatch.load(file_path=TEST_FOLDER / "mock_multibatch.json")
        self.multibatch.analysis_sum()
        self.vis = Visualizer(self.task, self.dataset, save=True)
        print(self.multibatch.batches)

    @patch.object(alt.Chart, "save")
    def test_stacked_bar(self, mock_save):
        chart = self.vis.stacked_bar(self.batch.mean_and_std["mean"])
        self.assertEqual(chart.to_dict()["mark"]["type"], "bar")
        # self.assertTupleEqual(chart.data.shape, (168, 4))
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / f"{self.task.id}.html"), format="html"
        )

    @patch.object(alt.Chart, "save")
    def test_sensitivity_stacked_bar(self, mock_save):
        chart = self.vis.stacked_bar(
            self.multibatch.batches._prep_for_vis("D_min"), param=param
        )
        self.assertEqual(chart.to_dict()["mark"]["type"], "bar")
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / f"{self.task.id}.html"), format="html"
        )

    @patch.object(alt.Chart, "save")
    def test_boxplot(self, mock_save):
        chart = self.(self.multibatch.summed_analysis)
        self.assertIn("hconcat", chart.to_dict())
        # self.assertTupleEqual(chart.data.shape, (36, 6))
        # mock_save.assert_called_with(
        #     str(OUTPUT_FOLDER / f"{self.task.id}_sensitivity.html"), format="html"
        # )
