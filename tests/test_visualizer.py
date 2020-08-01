import unittest
from unittest.mock import patch
import pandas as pd
import pandas.testing as pdt
import altair as alt

from simulation.output import Output
from simulation.sensitivity_analysis import SensitivityOutput
from simulation.dataset import Dataset
from simulation.visualizer import Visualizer
from simulation.task import Task
from simulation.constants import *


class TestVisualizer(unittest.TestCase):
    @patch.object(pd.DataFrame, "to_csv")
    def setUp(self, mock_to_csv):
        self.dataset = Dataset("mock_data")
        self.task = Task()
        self.output = Output(self.dataset, self.task)
        self.output.average = pd.read_csv(
            Path(TEST_FOLDER / "mock_output_averaged.csv")
        )
        self.output.concated = pd.concat([self.output.average * 2])
        self.output.export(how="average")
        self.vis = Visualizer(self.output, self.task, self.dataset)

    @patch.object(alt.Chart, "save")
    def test_visualizer(self, mock_save):
        chart = self.vis.visualize()
        self.assertEqual(chart.to_dict()["mark"], "bar")
        pdt.assert_frame_equal(chart.data, self.output.average)
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / f"{self.task.id}.html"), format="html"
        )

    @patch.object(alt.FacetChart, "save")
    def test_variance_boxplot(self, mock_save):
        chart = self.vis.variance_boxplot()
        self.assertEqual(chart.to_dict()["spec"]["mark"], "boxplot")
        pdt.assert_frame_equal(chart.data, self.output.concated)
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / f"{self.task.id}_variance.html"), format="html"
        )

    @patch.object(alt.FacetChart, "save")
    def test_sensitivity_boxplot(self, mock_save):
        self.output = SensitivityOutput(self.dataset, self.task)
        self.output.results = [
            pd.DataFrame(
                {
                    "peak_infected": [1, 2, 3],
                    "step": ["-1"] * 3,
                    "parameter": ["D_min"] * 3,
                }
            )
        ] * 10

        self.vis = Visualizer(output=self.output, task=self.task, dataset=self.dataset)
        chart = self.vis.sensitivity_boxplot(grouping="step")
        # chart = self.vis.sensitivity_boxplot(grouping="parameter")
        self.assertEqual(chart.to_dict()["spec"]["mark"], "boxplot")
        pdt.assert_frame_equal(chart.data, pd.concat(self.output.results))
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / f"{self.task.id}_sensitivity.html"), format="html"
        )
