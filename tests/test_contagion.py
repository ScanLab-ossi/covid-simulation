import unittest
from datetime import date
import os
import pandas as pd
import pandas.testing as pdt
import numpy as np

from simulation.simulation import test_conf
from simulation.contagion import CSVContagion, SQLContagion, Contagion
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.basic_configuration import BasicConfiguration
from simulation.constants import SKIP_TESTS


class TestContagion(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        bc = BasicConfiguration()
        dataset.load_dataset(gcloud=GoogleCloud(bc))
        self.c = Contagion(dataset=dataset, task_conf=test_conf)
        self.sample_infected = pd.DataFrame(
            {"duration": [1440, 36, 0], "color": [True, False, False]},
            index=[".QP/64EdoTcdkMnmXGVO0A"] * 3,
        )
        self.c.task_conf.update(
            {"D_min": 2, "D_max": 1440, "P_max": 0.8, "alpha_blue": 0.5}
        )

    def test_is_above_threshold(self):
        self.assertTrue(self.c._is_above_threshold(pd.Series([0.1])).all())

    def test_cases(self):
        self.c.task_conf["infection_model"] = 1
        piped = self.sample_infected.pipe(self.c._cases)
        should_be = self.sample_infected.copy()
        should_be["duration"] = [0.8, 0.02, 0]
        pdt.assert_frame_equal(piped, should_be)
        self.c.task_conf["infection_model"] = 2
        piped = self.sample_infected.pipe(self.c._cases)
        pdt.assert_frame_equal(piped, self.sample_infected.replace(0, 0.00001))

    def test_multiply_not_infected_chances(self):
        self.assertEqual(
            self.c._multiply_not_infected_chances(self.sample_infected["duration"]),
            0.804,
        )

    def test_consider_alpha(self):
        pdt.assert_frame_equal(
            self.c._consider_alpha(self.sample_infected),
            self.sample_infected.replace(36, 18),
        )


class TestCSVContagion(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        bc = BasicConfiguration()
        dataset.load_dataset(gcloud=GoogleCloud(bc))
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
            {"color": [True]}, index=["D8hZWX/ycJMmF4qg1uGkZc"]
        )
        self.cc.task_conf.update(
            {"D_min": 2, "D_max": 1440, "P_max": 0.8, "alpha_blue": 0.5}
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
        for i in range(1, 3):
            self.cc.task_conf["infection_model"] = i
            result = self.cc.contagion(self.sample_df, date(2012, 3, 28))
            self.assertEqual(result.iloc[0].name, "qj.1cdckhagf31nPKX0UjI")


@unittest.skipIf(SKIP_TESTS, "Skip SQL Tests")
class TestSQLContagion(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("h3g")
        self.sc = SQLContagion(dataset=dataset, task_conf=test_conf)
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
            len(self.sc.contagion(self.potential_patients, "2012-03-29")), 7942,
        )
