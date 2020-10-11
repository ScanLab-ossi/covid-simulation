import unittest
import pandas as pd
import pandas.testing as pdt
import numpy as np
from datetime import date
from unittest.mock import patch


from simulation.output import Output
from simulation.state_transition import StateTransition
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.task import Task


class TestStateTransition(unittest.TestCase):
    def setUp(self):
        task = Task()
        dataset = Dataset("mock_data")
        dataset.load_dataset(GoogleCloud())
        self.output = Output(dataset, task)
        self.st = StateTransition(dataset, task)
        self.sample_infected = pd.DataFrame(
            [[10000]], columns=["duration"], index=[".QP/64EdoTcdkMnmXGVO0A"]
        )
        self.st.task.update(
            {"D_min": 2, "D_max": 1440, "P_max": 0.8, "threshold": 0.05, "S_i": 0.7}
        )

    # @patch("numpy.random.rand")
