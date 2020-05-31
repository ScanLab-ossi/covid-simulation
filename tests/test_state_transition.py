import unittest
import pandas as pd
import numpy as np
from datetime import date

from simulation.output import Output
from simulation.state_transition import StateTransition
from simulation.dataset import Dataset
from simulation.simulation import test_conf


class TestStateTransition(unittest.TestCase):
    def setUp(self):
        dataset = Dataset("mock_data")
        dataset.load_dataset()
        self.output = Output(dataset)
        self.st = StateTransition(dataset=dataset, task_conf=test_conf)
        self.sample_infected = pd.DataFrame(
            [[10000]], columns=["daily_duration"], index=[".QP/64EdoTcdkMnmXGVO0A"]
        )

    def test_check_if_aggravation(self):
        pass
        # np.random.seed(seed=2)  # prob = 0.4
        # self.assertEqual(d.check_if_aggravate(np.arange(10)).all(), False)
        # self.assertEqual(d.check_if_aggravate(np.arange(10), s_i=0.1).all(), True)

    # def test_check_if_infected(self):
    #     self.assertTrue(sim.Data.check_if_infected(1, 1))
    #     self.assertTrue(sim.Data.check_if_infected(0.5, 0.5))
    #     self.assertTrue(sim.Data.check_if_infected(0.1, 1) is True)
    #     self.assertTrue(sim.Data.check_if_infected(0.1, 1, 0.05))

    def test_infection_state_transition_return_values(self):
        df = self.st._infection_state_transition(
            self.sample_infected, date(2012, 3, 26)
        )
        self.assertEqual(
            set(df["age_group"]) - set(range(len(test_conf["age_dist"]))), set()
        )
        self.assertTrue(df["color"].isin(np.array([True, False])).all())
        self.assertTrue((df["expiration_date"].values > date(2012, 3, 26)).all())

    def test_is_enough_duration(self):
        self.assertEqual(
            self.st._is_enough_duration(self.sample_infected["daily_duration"]),
            np.array([True]),
        )
        self.assertEqual(
            self.st._is_enough_duration(
                self.sample_infected["daily_duration"].replace(10000, 10)
            ),
            np.array([False]),
        )

    def test_get_trajectory(self):
        self.st.get_trajectory(
            self.sample_infected,
            self.output,
            self.st.dataset.start_date,
            add_duration=False,
        )
        self.assertListEqual(
            self.output.df.columns.tolist(),
            [
                "age_group",
                "color",
                "infection_date",
                "transition_date",
                "expiration_date",
                "final_state",
                "daily_duration",
            ],
        )
        self.assertEqual(self.output.df.index.name, "id")


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
