import unittest
import pandas as pd

from simulation.output import Output


class TestOutput(unittest.TestCase):
    def setUp(self):
        self.output = Output()

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
