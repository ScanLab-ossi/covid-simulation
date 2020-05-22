import unittest
import configparser, os
from pathlib import Path
from unittest import mock

from simulation.basic_configuration import BasicConfiguration


class TestBasicConfiguration(unittest.TestCase):
    def setUp(self):
        self.bc = BasicConfiguration()
        # os.path.exists = MockPathExists(True)

    def test_get_config(self):
        self.assertIsInstance(self.bc.get_config(), configparser.ConfigParser)

    @mock.patch.dict(
        os.environ,
        {
            "POSTGRES_DATABASE_NAME": "test",
            "POSTGRES_USER_NAME": "test",
            "POSTGRES_USER_PASSWORD": "test",
            "POSTGRES_DATABASE_HOST": "test",
        },
    )
    def test_conf(self):
        self.assertSetEqual(
            set(self.bc.config["postgres"].keys()),
            {"dbname", "user", "password", "host"},
        )
        self.assertIsNotNone(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None))
        self.assertTrue(Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).exists())
