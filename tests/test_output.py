import unittest
import pandas as pd
from datetime import date
import pandas.testing as pdt
from unittest.mock import patch

from simulation.output import Output
from simulation.dataset import Dataset
from simulation.constants import *
from simulation.task import Task


class TestOutput(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        self.task = Task()
        self.output = Output(dataset, self.task)
        self.sample_df = pd.DataFrame(
            [
                [0, True, date(2012, 3, 26), date(2012, 4, 1), date(2012, 4, 1), "w"],
                [2, False, date(2012, 3, 27), date(2012, 4, 5), date(2012, 4, 7), "w"],
            ],
            columns=[
                "age_group",
                "color",
                "infection_date",
                "transition_date",
                "expiration_date",
                "final_state",
            ],
            index=["FNGiD7T4cpkOIM3mq.YdMY", "ODOkY9pchzsDHj.23UGQoc"],
        )
        self.sample_empty_df = self.df = pd.DataFrame(
            columns=[
                "age_group",
                "color",
                "infection_date",
                "transition_date",
                "expiration_date",
                "final_state",
            ]
        )
        self.sample_empty_df.index.name = "id"
        self.sample_summed_df = pd.DataFrame(
            [["b", 0, 10.0], ["g", 0, 10.0]], columns=["color", "day", "amount"]
        )

    def test_reset(self):
        pdt.assert_frame_equal(self.output.df, self.sample_empty_df)
        self.output.df.append(self.sample_df)
        self.output.reset()
        pdt.assert_frame_equal(self.output.df, self.sample_empty_df)

    # @patch("builtins.open", new_callable=mock_open)
    @patch.object(pd.DataFrame, "to_pickle")
    @patch.object(pd.DataFrame, "to_csv")
    def test_export(self, mock_to_csv, mock_to_pickle):
        with self.assertRaises(AttributeError):
            self.output.export(how="test")
        self.output.average = self.sample_df
        self.output.batch.append(self.sample_df)
        self.output.export(how="average", pickle=True)
        mock_to_pickle.assert_called_once_with(
            Path(OUTPUT_FOLDER / f"{self.task.id}.pkl")
        )
        mock_to_csv.assert_called_once_with(
            Path(OUTPUT_FOLDER / f"{self.task.id}.csv"), index=False
        )

    def test_append_row_to_df(self):
        self.assertEqual(len(self.output.df), 0)
        self.output.append(self.sample_df)
        self.assertEqual(len(self.output.df), 2)
        with self.assertRaises(ValueError):
            self.output.append(self.sample_df)

    def test_sum_output(self):
        summed = self.output.sum_output(self.sample_df)
        self.assertEqual(summed.shape, (396, 3))  # 66 days in mockdata * 6 colors
        self.assertEqual(summed.columns.tolist(), ["color", "day", "amount"])

    def test_average_outputs(self):
        self.output.concated = pd.concat([self.sample_summed_df] * 3)
        self.output.average_outputs()
        pdt.assert_frame_equal(self.sample_summed_df, self.output.average)
