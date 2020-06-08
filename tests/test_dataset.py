import unittest

from simulation.simulation import test_conf
from simulation.contagion import CSVContagion
from simulation.dataset import Dataset
from simulation.helpers import timing
from simulation.basic_configuration import BasicConfiguration
from simulation.google_cloud import GoogleCloud


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset("mock_data")
        bc = BasicConfiguration()
        self.dataset.load_dataset(gcloud=GoogleCloud(bc))

    def test_dataset_structure(self):
        self.assertEqual(
            set(self.dataset.data.columns)
            - {"source", "destination", "datetime", "distance", "duration"},
            set(),
        )

    def test_attributes(self):
        # TODO: test structure of datasets.json
        pass
