import unittest
import pandas as pd
from datetime import date

from simulation.output import Output
from simulation.dataset import Dataset


class TestOutput(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        self.output = Output(dataset=dataset)

    def test_create_empty_dataframe(self):
        self.assertListEqual(
            self.output.df.columns.tolist(),
            ["age_group", "color", "infection_date", "expiration_date"],
        )
        self.assertEqual(self.output.df.index.name, "id")

    def test_append_row_to_df(self):
        self.assertEqual(len(self.output.df), 0)
        sample_df = pd.DataFrame(
            [range(len(self.output.df.columns))],
            columns=self.output.df.columns,
            index=["sample_id"],
        )
        self.output.append(sample_df)
        self.assertEqual(len(self.output.df), 1)
        with self.assertRaises(ValueError):
            self.output.append(sample_df)

    def test_sum_output(self):
        sample_df = pd.DataFrame(
            [
                [0, True, date(2012, 3, 26), date(2012, 4, 1)],
                [2, False, date(2012, 3, 27), date(2012, 4, 5)],
            ],
            columns=["age_group", "color", "infection_date", "expiration_date"],
            index=["FNGiD7T4cpkOIM3mq.YdMY", "ODOkY9pchzsDHj.23UGQoc"],
        )
        self.output.append(sample_df)
        summed = self.output.sum_output()
        self.assertEqual(summed.shape, (330, 3))
        self.assertEqual(summed.columns.tolist(), ["color", "day", "amount"])
