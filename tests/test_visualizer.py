import unittest
from unittest.mock import patch
import pandas as pd
import pandas.testing as pdt
import altair as alt

from simulation.output import Batch, MultiBatch
from simulation.dataset import Dataset
from simulation.visualizer import Visualizer
from simulation.task import Task
from simulation.constants import *


class TestVisualizer(unittest.TestCase):
    @patch.object(pd.DataFrame, "to_csv")
    def setUp(self, mock_to_csv):
        self.dataset = Dataset("copenhagen_hops")
        self.task = Task()
        self.batch = Batch(self.task)
        self.batch.load(TEST_FOLDER / "mock_summed_batch.csv")
        self.multibatch = MultiBatch(self.task)
        self.multibatch.load(TEST_FOLDER / "mock_summed_multibatch.csv")

    @patch.object(alt.Chart, "save")
    def test_visualizer(self, mock_save):
        self.batch.average_outputs()
        vis = Visualizer(self.task, self.dataset, self.batch, save=True)
        chart = vis.visualize()
        self.assertEqual(chart.to_dict()["mark"]["type"], "bar")
        # self.assertTupleEqual(chart.data.shape, (168, 4))
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / f"{self.task.id}.html"), format="html"
        )

    @unittest.skipIf(True, "currently not supported")
    @patch.object(alt.FacetChart, "save")
    def test_variance_boxplot(self, mock_save):
        chart = self.vis.variance_boxplot()
        self.assertEqual(chart.to_dict()["spec"]["mark"], "boxplot")
        pdt.assert_frame_equal(chart.data, self.output.concated)
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / f"{self.task.id}_variance.html"), format="html"
        )

    @patch.object(alt.Chart, "save")
    def test_concated_boxplots(self, mock_save):
        vis = Visualizer(self.task, self.dataset, self.multibatch, save=True)
        chart = vis.concated_boxplots()
        self.assertIn("hconcat", chart.to_dict())
        # self.assertTupleEqual(chart.data.shape, (36, 6))
        # mock_save.assert_called_with(
        #     str(OUTPUT_FOLDER / f"{self.task.id}_sensitivity.html"), format="html"
        # )
