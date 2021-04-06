import unittest
from unittest.mock import patch
import _pickle as cPickle
from datetime import date
import json

import pandas as pd
import numpy as np
import pandas.testing as pdt

from simulation.output import Output, Batch, MultiBatch
from simulation.dataset import Dataset
from simulation.constants import *
from simulation.task import Task

output_columns = [
    "age_group",
    "color",
    "infection_date",
    "transition_date",
    "expiration_date",
    "final_state",
]
sample_df = pd.DataFrame(
    [
        [0, True, date(2012, 3, 26), date(2012, 4, 1), date(2012, 4, 1), "w"],
        [2, False, date(2012, 3, 27), date(2012, 4, 5), date(2012, 4, 7), "w"],
    ],
    columns=output_columns,
    index=["FNGiD7T4cpkOIM3mq.YdMY", "ODOkY9pchzsDHj.23UGQoc"],
)
sample_empty_df = pd.DataFrame(columns=output_columns).rename_axis("id", axis="index")
sample_summed_df = pd.DataFrame(
    [["b", 0, 10.0], ["g", 0, 10.0]], columns=["color", "day", "amount"]
)


class TestOutput(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        task = Task(test=True)
        self.output = Output(dataset, task)

    def test_len(self):
        self.output.df = sample_df
        self.assertEqual(len(self.output), 2)

    def test_append_row_to_df(self):
        self.assertEqual(len(self.output), 0)
        self.output.df = sample_df
        self.assertEqual(len(self.output), 2)
        # with self.assertRaises(ValueError):
        #     self.output.df.append(sample_df)

    def test_sum_output(self):
        # TODO:
        # self.assertEqual(summed.shape, (396, 3))  # 66 days in mockdata * 6 colors
        # self.assertEqual(summed.columns.tolist(), ["color", "day", "amount"])
        pass


class TestBatch(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        task = Task(test=True)
        self.batch = Batch(dataset, task)
        for df in np.array_split(
            pd.read_csv(TEST_FOLDER / "mock_summed_batch.csv").groupby("day"), 3
        ):
            o = Output(dataset, task)
            o.df = df
            self.batch.append_output(o)

    @patch.object(cPickle, "dump")
    @patch.object(json, "dump")
    def test_export(self, mock_to_json, mock_to_pickle):
        with self.assertRaises(AttributeError):
            self.output.export(how="test")
        self.average = sample_df
        self.batch.sum_batch()
        # self.batch.export(what="summed", format_="pickle")
        self.batch.export("mean_and_std")
        # mock_to_pickle.assert_called_once_with(OUTPUT_FOLDER / f"{self.task.id}.pkl")
        mock_to_json.assert_called_once()
