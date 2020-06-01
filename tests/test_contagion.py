import unittest
from datetime import date
import os
import pandas as pd

from simulation.simulation import test_conf
from simulation.contagion import CSVContagion, SQLContagion
from simulation.dataset import Dataset
from simulation.constants import SKIP_TESTS


class TestCSVContagion(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        dataset.load_dataset()
        self.cc = CSVContagion(dataset=dataset, task_conf=test_conf)
        self.potential_patients = {
            ".QP/64EdoTcdkMnmXGVO0A",
            "BP51jL2myIMRqfYseLbGfM",
            "D8hZWX/ycJMmF4qg1uGkZc",
            "FNGiD7T4cpkOIM3mq.YdMY",
            "HhulO23UWA2BVHqsECvjJY",
            "Sq1s6KEGp1Qm8MN1o1paM.",
            "cMvEW1y.DLUsMgtP951/f.",
            "czqASEiMg7MUBvYLidDHZY",
            "m5YJXcVamIkxaZmrDw1mwA",
            "xDK0mIGasmAilJrvnFS3Pw",
        }
        self.sample_df = pd.DataFrame(
            [[True], [False]],
            columns=["color"],
            index=["..7cyMMPqV.bMVjsN7Rcns", "..cvdr3nnY2eZmwko9evCQ"],
        )

    def test_patient_zero(self):
        self.assertEqual(
            self.cc.pick_patient_zero(self.potential_patients)
            - self.potential_patients,
            set(),
        )
        self.assertEqual(len(self.cc.pick_patient_zero(self.potential_patients)), 10)

    def test_if_patient_zero_arbitrarily_selected(self):
        self.assertSetEqual(
            self.cc.pick_patient_zero(
                None, arbitrary_patient_zero=["MJviZSTPuYw1v0W0cURthY"]
            ),
            {"MJviZSTPuYw1v0W0cURthY"},
        )

    def test_first_circle_of_2_patient_in_specific_date(self):
        result = self.cc.contagion(self.sample_df, date(2012, 3, 26))
        self.assertEqual(result.iloc[0].name, "cMvEW1y.DLUsMgtP951/f.")


@unittest.skipIf(SKIP_TESTS, "Skip SQL Tests")
class TestSQLContagion(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("h3g")
        self.cc = SQLContagion(dataset=dataset, task_conf=test_conf)
        self.potential_patients = {
            ".QP/64EdoTcdkMnmXGVO0A",
            "BP51jL2myIMRqfYseLbGfM",
            "D8hZWX/ycJMmF4qg1uGkZc",
            "FNGiD7T4cpkOIM3mq.YdMY",
            "HhulO23UWA2BVHqsECvjJY",
            "Sq1s6KEGp1Qm8MN1o1paM.",
            "cMvEW1y.DLUsMgtP951/f.",
            "czqASEiMg7MUBvYLidDHZY",
            "m5YJXcVamIkxaZmrDw1mwA",
            "xDK0mIGasmAilJrvnFS3Pw",
        }

    def test_sql_query_contagion(self):
        self.assertEqual(
            len(self.contagion(self.potential_patients, "2012-03-29")), 7942,
        )
