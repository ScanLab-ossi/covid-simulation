import unittest
import pandas as pd
import pandas.testing as pdt
import numpy as np
from datetime import date
from unittest.mock import patch


from simulation.output import Output
from simulation.state_transition import StateTransition
from simulation.dataset import Dataset
from simulation.basic_configuration import BasicConfiguration
from simulation.google_cloud import GoogleCloud
from simulation.task import Task


class TestStateTransition(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        task = Task()
        bc = BasicConfiguration()
        dataset.load_dataset(GoogleCloud())
        self.output = Output(dataset, task)
        self.st = StateTransition(dataset, task)
        self.sample_infected = pd.DataFrame(
            [[10000]], columns=["duration"], index=[".QP/64EdoTcdkMnmXGVO0A"]
        )
        self.st.task.update(
            {"D_min": 2, "D_max": 1440, "P_max": 0.8, "threshold": 0.05, "S_i": 0.7}
        )

    @patch("numpy.random.rand")
    def test_check_if_aggravate(self, rand_mock):
        rand_mock.return_value = np.array([0.8] * 3)
        self.assertTrue(self.st._check_if_aggravate(np.array([1, 2, 3])).all())

    @patch("numpy.random.normal")
    def test_get_next_date(self, mock_normal):
        self.assertEqual(
            self.st._get_next_date(("same", date(2012, 3, 26))), date(2012, 3, 26)
        )
        mock_normal.return_value = 15
        self.assertEqual(
            self.st._get_next_date(("blue_to_white", date(2012, 3, 26))),
            date(2012, 4, 10),
        )
        self.assertEqual(
            self.st._get_next_date(("blue_to_white", date(2012, 5, 31))),
            date(2012, 5, 31),
        )

    @patch("numpy.random.normal")
    def test_final_state(self, mock_normal):
        mock_normal.return_value = 1.0
        self.assertEqual(self.st._final_state(True), "k")
        self.assertEqual(self.st._final_state(False), "w")

    def test_infection_state_transition_return_values(self):
        df = self.st._infection_state_transition(
            self.sample_infected, date(2012, 3, 26)
        )
        self.assertEqual(
            set(df["age_group"]) - set(range(len(self.st.task["age_dist"]))), set()
        )
        self.assertTrue(df["color"].isin(np.array([True, False])).all())
        # infection_date, transition_date, final_state
        self.assertTrue((df["infection_date"].values == date(2012, 3, 26)).all())
        self.assertTrue((df["transition_date"].values > date(2012, 3, 26)).all())
        self.assertTrue((df["expiration_date"].values > date(2012, 3, 26)).all())
        self.assertTrue(df["final_state"].isin(np.array(["w", "k"])).all())
        dtypes = pd.Series(
            {
                "age_group": np.dtype("int64"),
                "color": np.dtype("bool"),
                "infection_date": np.dtype("O"),
                "transition_date": np.dtype("O"),
                "expiration_date": np.dtype("O"),
                "final_state": np.dtype("O"),
            }
        )
        pdt.assert_series_equal(df.dtypes, dtypes)

    def test_get_trajectory(self):
        for starter in [self.sample_infected, {".QP/64EdoTcdkMnmXGVO0A"}]:
            infected = self.st.get_trajectory(starter, self.output, date(2012, 3, 26))
            self.assertListEqual(
                infected.columns.tolist(),
                (["duration"] if isinstance(starter, pd.DataFrame) else [])
                + [
                    "age_group",
                    "color",
                    "infection_date",
                    "transition_date",
                    "expiration_date",
                    "final_state",
                ],
            )
            self.assertEqual(self.output.df.index.name, "id")
        # check that people don't get reinfected
        self.output.append(infected)
        reinfected = self.st.get_trajectory(
            {".QP/64EdoTcdkMnmXGVO0A"}, self.output, date(2012, 3, 26)
        )
        pdt.assert_frame_equal(reinfected, pd.DataFrame([]))


# def test_time_to_recovery(self):
#     default_config = sim.test_conf
#     np.random.seed(seed=1)
#     self.assertEqual(
#         sim.Data.time_to_recovery("2012-04-02", default_config), "2012-05-08"
#     )

# def test_time_to_aggravation(self):
#     default_config = sim.test_conf
#     np.random.seed(seed=2)
#     self.assertEqual(
#         sim.Data.time_to_aggravation("2012-04-02", default_config), "2012-04-06"
#     )
