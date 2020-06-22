import unittest
from unittest.mock import MagicMock
from google.cloud import storage, datastore
from google.cloud import exceptions as gcloud_exceptions
from pathlib import Path

from simulation.google_cloud import GoogleCloud
from simulation.task import Task
from simulation.basic_configuration import BasicConfiguration
from simulation.constants import settings


class GoogleCloudTest(unittest.TestCase):
    # TODO: add mock
    def setUp(self):
        bc = BasicConfiguration()
        self.gcloud = GoogleCloud(bc)

    def test_connection_to_google_cloud_storage(self):
        bucket = self.gcloud.s_client.lookup_bucket("simulation_runs")
        self.assertIsInstance(bucket, storage.Bucket)

    @unittest.skipIf(settings["SKIP_TESTS"], "Skip Google Storage Tests")
    def test_download(self):
        self.gcloud.download("test.csv")
        self.assertTrue(Path("./data/test.csv").exists())

    @unittest.skipIf(settings["SKIP_TESTS"], "Skip Google Storage Tests")
    def test_upload(self):
        bucket = self.gcloud.s_client.bucket("simulation_datasets")
        blob = bucket.blob("test.csv")
        try:
            blob.delete()
            print("deleted")
        except gcloud_exceptions.NotFound:
            pass
        url = self.gcloud.upload(
            Path("./data/test.csv"), bucket_name="simulation_datasets"
        )
        self.assertEqual(
            url,
            "https://www.googleapis.com/storage/v1/b/simulation_datasets/o/test.csv",
        )
        self.assertTrue(blob.exists())

    def test_connection_to_google_cloud_datastore(self):
        pass

    def test_get_tasklist(self):
        self.gcloud.get_tasklist()
        all_tasks = self.gcloud.todo + self.gcloud.done
        self.assertNotIn(False, [isinstance(t, Task) for t in all_tasks])

    @unittest.skipIf(settings["SKIP_TESTS"], "Skip Google Storage Tests")
    def test_add_task(self):
        self.gcloud.ds_client.put = MagicMock(return_value=True)
        self.assertTrue(self.gcloud.add_task({"dataset": "h3g"}, Task()))
