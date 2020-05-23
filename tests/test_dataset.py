import unittest

from simulation.simulation import test_conf
from simulation.contagion import CSVContagion
from simulation.dataset import Dataset
from simulation.helpers import timing


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset("mock_data")
        self.dataset.load_dataset()

    def test_dataset_structure(self):
        self.assertEqual(
            set(self.dataset.data.columns)
            - {"source", "destination", "datetime", "distance", "duration"},
            set(),
        )

    def test_attributes(self):
        # TODO: test structure of datasets.json
        pass
