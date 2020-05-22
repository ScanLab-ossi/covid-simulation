import unittest
import numpy as np

from simulation.task_config import TaskConfig

test_conf = {
    "age_dist": np.array([0.15, 0.6, 0.25]),  # [youngs, adults, olds]
    "recovery_time_dist": np.array(
        [20, 10]
    ),  # recovery_time_dist ~ Norm(mean, std) | ref:
    "aggravation_time_dist": np.array(
        [5, 2]
    ),  # aggravation_time_dist ~ Norm(mean, std) | ref:
    "D_min": 10,  # Arbitrary, The minimal threshold (in time) for infection,
    "number_of_patient_zero": 10,  # Arbitrary
    "D_max": 70,  # Arbitrary, TO BE CALCULATED,  0.9 precentile of (D_i)'s
    "P_max": 0.2,  # The probability to be infected when the exposure is over the threshold
    "risk_factor": None,  # should be vector of risk by age group
}


class TestTaskConfig(unittest.TestCase):
    def setUp(self):
        self.tc = TaskConfig(test_conf)

    def test_task_config_validator(self):
        self.assertTrue(self.tc.is_valid)

    def test_task_config_as_lists(self):
        self.assertNotIn(
            True, [isinstance(x, np.ndarray) for x in self.tc.as_lists().values()]
        )

    def test_task_config_as_ndarrays(self):
        self.assertNotIn(
            True, [isinstance(x, list) for x in self.tc.as_ndarrays().values()]
        )
