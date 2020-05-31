import unittest
from unittest.mock import patch
import pandas as pd
import pandas.testing as pdt
import altair as alt

from simulation.output import Output
from simulation.dataset import Dataset
from simulation.visualizer import Visualizer
from simulation.constants import *


class TestOutput(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        self.output = Output(dataset=dataset)
        self.output.average = pd.read_csv(Path(OUTPUT_FOLDER / "mock_output.csv"))
        self.output.concated = pd.concat([self.output.average * 2])
        self.vis = Visualizer(self.output)

    @patch.object(alt.Chart, "save")
    def test_visualizer(self, mock_save):
        chart = self.vis.visualize()
        self.assertEqual(chart.to_dict()["mark"], "bar")
        pdt.assert_frame_equal(chart.data, self.output.average)
        mock_save.assert_called_with(str(OUTPUT_FOLDER / "output.html"), format="html")

    @patch.object(alt.FacetChart, "save")
    def test_boxplot_variance(self, mock_save):
        chart = self.vis.boxplot_variance()
        self.assertEqual(chart.to_dict()["spec"]["mark"], "boxplot")
        pdt.assert_frame_equal(chart.data, self.output.concated)
        mock_save.assert_called_with(
            str(OUTPUT_FOLDER / "output_variance.html"), format="html"
        )
