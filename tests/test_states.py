from simulation.states import States
from simulation.constants import TEST_FOLDER
from simulation.task import Task
import unittest

import pandas as pd
import pandas.testing as pdt


task = Task(path=TEST_FOLDER / "config.yaml")


class TestStates(unittest.TestCase):
    def setUp(self):
        self.states = States(task)
