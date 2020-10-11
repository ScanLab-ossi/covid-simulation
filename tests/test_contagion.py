import unittest
from datetime import date
import os
import pandas as pd
import pandas.testing as pdt
import numpy as np
import numpy.testing as npt

from simulation.contagion import CSVContagion, SQLContagion, Contagion
from simulation.task import Task
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.constants import settings

task = Task()
dataset = Dataset("mock_data")
dataset.load_dataset(gcloud=GoogleCloud())


class TestContagion(unittest.TestCase):
    def setUp(self):
        self.c = Contagion(dataset, task, reproducible=True)
        self.sample_infected = pd.DataFrame(
            {"duration": [1440, 36, 0], "color": [True, False, False]},
            index=[".QP/64EdoTcdkMnmXGVO0A"] * 3,
        )
        self.c.task.update({"D_min": 2, "D_max": 1440, "P_max": 0.8, "alpha_blue": 0.5})

    def test_cases(self):
        self.c.task["infection_model"] = 1
        piped = self.sample_infected.pipe(self.c._cases)
        should_be = self.sample_infected.copy()
        should_be["duration"] = [0.8, 0.02, 0]
        pdt.assert_frame_equal(piped, should_be)
        self.c.task["infection_model"] = 2
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

    def test_is_infected(self):
        npt.assert_array_equal(
            self.c._is_infected(pd.Series([1.0, 0.0])), np.array([True, False]),
        )


class TestCSVContagion(unittest.TestCase):
    def setUp(self):
        self.cc = CSVContagion(dataset, task, reproducible=True)
        self.potential_patients = {
            ".QP/64EdoTcdkMnmXGVO0A",
            "BP51jL2myIMRqfYseLbGfM",
            "D8hZWX/ycJMmF4qg1uGkZc",
            "FNGiD7T4cpkOIM3mq.YdMY",
            "HhulO23UWA2BVHqsECvjJY",
            "Sq1s6KEGp1Qm8MN1o1paM.",
            "czqASEiMg7MUBvYLidDHZY",
            "m5YJXcVamIkxaZmrDw1mwA",
            "2Onqyq2Ke1gkX5taqmTLCI",
            "xDK0mIGasmAilJrvnFS3Pw",
        }
        self.sample_df = pd.DataFrame(
            {"color": [True]}, index=["D8hZWX/ycJMmF4qg1uGkZc"]
        )
        self.cc.task.update(
            {
                "D_min": 0,
                "D_max": 1440,
                "P_max": 0.8,
                "alpha_blue": 1,
                "number_of_patient_zero": 1,
            }
        )
        self.cc.task.update

    def test_patient_zero(self):
        print(f"ppz = {self.cc.pick_patient_zero()}")
        res = pd.DataFrame(
            [0, 0, "green"],
            columns=["infection_date", "days_left", "color"],
            index=["eabfCPikoZg8D1UxBq4NnA"],
        )
        pdt.assertFrameEqual(self.cc.pick_patient_zero(), res)

    # def test_first_circle_of_2_patient_in_specific_date(self):
    #     for i in [1, 2]:
    #         self.cc.task["infection_model"] = i
    #         ddd = pd.DataFrame(
    #             [[True]] * len(self.potential_patients),
    #             columns=["color"],
    #             index=self.potential_patients,
    #         )
    #         result = self.cc.contagion(ddd, date(2012, 3, 26))
    #         print(result)
    #         self.assertEqual(result.iloc[0].name, "qj.1cdckhagf31nPKX0UjI")


@unittest.skipIf(settings["SKIP_TESTS"], "Skip SQL Tests")
class TestSQLContagion(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("h3g")
        self.sc = SQLContagion(dataset=dataset, task=task)
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
