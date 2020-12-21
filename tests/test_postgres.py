import unittest
import configparser, os
from pathlib import Path
from unittest import mock

from simulation.postgres import Postgres
from simulation.google_cloud import GoogleCloud


class TestPostgres(unittest.TestCase):
    def setUp(self):
        self.postgres = Postgres(GoogleCloud())
        # os.path.exists = MockPathExists(True)

    def test_get_config(self):
        self.assertIsInstance(self.postgres.config, configparser.ConfigParser)
        # self.assertTrue(Path("./secrets.conf").exists())

    # @mock.patch.dict(
    #     os.environ,
    #     {
    #         "POSTGRES_DATABASE_NAME": "test",
    #         "POSTGRES_USER_NAME": "test",
    #         "POSTGRES_USER_PASSWORD": "test",
    #         "POSTGRES_DATABASE_HOST": "test",
    #     },
    # )
    # def test_conf(self):
    #     self.assertSetEqual(
    #         set(self.postgres.config["postgres"].keys()),
    #         {"dbname", "user", "password", "host"},
    #     )
