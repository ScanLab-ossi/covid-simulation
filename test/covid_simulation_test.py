import unittest
from unittest import mock
from unittest.mock import MagicMock
import numpy as np
from random import seed
from datetime import datetime, date
import pandas as pd
import os, configparser
from google.cloud import storage, datastore
from google.cloud import exceptions as gcloud_exceptions
from pathlib import Path

import simulation.simulation as sim

SKIP_TESTS = os.environ.get("CI_SKIP_TESTS", False)


class PickZeroPatients(unittest.TestCase):
    def test_if_patient_zero_default_selected(self):
        set_of_mock_potential_patients = {
            ".QP/64EdoTcdkMnmXGVO0A",
            "BP51jL2myIMRqfYseLbGfM,D8hZWX/ycJMmF4qg1uGkZc",
            " FNGiD7T4cpkOIM3mq.YdMY",
            "HhulO23UWA2BVHqsECvjJY",
            "Sq1s6KEGp1Qm8MN1o1paM.",
            "cMvEW1y.DLUsMgtP951/f.",
            "czqASEiMg7MUBvYLidDHZY",
            "m5YJXcVamIkxaZmrDw1mwA",
            "xDK0mIGasmAilJrvnFS3Pw",
        }
        self.assertEqual(len(sim.pick_patient_zero(set_of_mock_potential_patients)), 1)
        self.assertEqual(
            sim.pick_patient_zero(set_of_mock_potential_patients)[0]
            in set_of_mock_potential_patients,
            True,
        )

    def test_if_more_than_one_patient_zero_selected(self):
        set_of_mock_potential_patients = {
            ".QP/64EdoTcdkMnmXGVO0A",
            "BP51jL2myIMRqfYseLbGfM,D8hZWX/ycJMmF4qg1uGkZc",
            " FNGiD7T4cpkOIM3mq.YdMY",
            "HhulO23UWA2BVHqsECvjJY",
            "Sq1s6KEGp1Qm8MN1o1paM.",
            "cMvEW1y.DLUsMgtP951/f.",
            "czqASEiMg7MUBvYLidDHZY",
            "m5YJXcVamIkxaZmrDw1mwA",
            "xDK0mIGasmAilJrvnFS3Pw",
        }
        self.assertEqual(
            len(
                sim.pick_patient_zero(set_of_mock_potential_patients, num_of_patients=3)
            ),
            3,
        )

    def test_if_patient_zero_arbitrarily_selected(self):
        self.assertEqual(
            sim.pick_patient_zero(None, arbitrary=True,), ["MJviZSTPuYw1v0W0cURthY"]
        )

    def test_if_patient_zero_user_arbitrarily_selected(self):
        self.assertEqual(
            sim.pick_patient_zero(
                None, arbitrary=True, arbitrary_patient_zero=["JYiZSTPuYw1v0W0cURthY"]
            ),
            ["JYiZSTPuYw1v0W0cURthY"],
        )


class ReadingDataFromMock(unittest.TestCase):
    def test_get_active_ids_from_mock_file(self):
        self.assertEqual(len(sim.get_active_ids("./data/mock_data.csv")), 4814)


class SpreadOfTheDisease(unittest.TestCase):
    def test_first_circle_of_patient_in_specific_date(self):
        result = sim.contagion_in_csv(
            "./data/mock_data.csv", {"FNGiD7T4cpkOIM3mq.YdMY"}, date(2012, 3, 26)
        )
        self.assertEqual(result.iloc[0].name, "ODOkY9pchzsDHj.23UGQoc")

    def test_first_circle_of_2_patient_in_specific_date(self):
        result = sim.contagion_in_csv(
            "./data/mock_data.csv",
            {"..7cyMMPqV.bMVjsN7Rcns", "..cvdr3nnY2eZmwko9evCQ"},
            date(2012, 3, 26),
        )

        self.assertEqual(result.iloc[0].name, "cMvEW1y.DLUsMgtP951/f.")

    def test_virus_spread_over_some_days(self):
        patient_set = {
            "FNGiD7T4cpkOIM3mq.YdMY",
            "Sq1s6KEGp1Qm8MN1o1paM.",
            "HhulO23UWA2BVHqsECvjJY",
        }
        result = sim.virus_spread(
            "./data/mock_data.csv", patient_set, date(2012, 4, 2), 2
        )
        self.assertEqual(len(result), 12)


class TestBasicConfiguration(unittest.TestCase):
    # def setUp(self):
    #     os.path.exists = MockPathExists(True)

    def test_get_config(self):
        pass
        conf = sim.BasicConfiguration()
        self.assertIsInstance(conf.get_config(), configparser.ConfigParser)

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
        conf = sim.BasicConfiguration()
        self.assertSetEqual(
            set(conf.config["postgres"].keys()), {"dbname", "user", "password", "host"},
        )
        self.assertIsNotNone(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None))
        self.assertTrue(Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).exists())


class TestTaskConfig(unittest.TestCase):
    def test_task_config_validator(self):
        tc = sim.TaskConfig(sim.test_conf)
        self.assertTrue(tc.is_valid)

    def test_task_config_as_lists(self):
        tc = sim.TaskConfig(sim.test_conf)
        self.assertNotIn(
            True, [isinstance(x, np.ndarray) for x in tc.as_lists().values()]
        )

    def test_task_config_as_ndarrays(self):
        tc = sim.TaskConfig(sim.test_conf)
        self.assertNotIn(True, [isinstance(x, list) for x in tc.as_ndarrays().values()])


class DefaultTaskConfig(unittest.TestCase):
    def test_default_configuration(self):
        default_config = sim.test_conf
        self.assertEqual(
            (default_config.get("age_dist") == np.array([0.15, 0.6, 0.25])).all(), True,
        )
        self.assertEqual(
            (default_config.get("recovery_time_dist") == np.array([20, 10])).all(),
            True,
        )
        self.assertEqual(
            (default_config.get("aggravation_time_dist") == np.array([5, 2])).all(),
            True,
        )
        self.assertEqual(default_config.get("D_min"), 10)
        self.assertEqual(default_config.get("number_of_patient_zero"), 10)
        self.assertEqual(default_config.get("D_max"), 70)


class DataStructure(unittest.TestCase):
    def test_create_empty_dataframe(self):
        d = sim.Data()
        self.assertListEqual(
            d.df.columns.tolist(),
            ["age_group", "color", "infection_date", "expiration_date"],
        )
        self.assertEqual(d.df.index.name, "id")
        # d.display()

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

    def test_check_if_aggravation(self):
        pass
        d = sim.Data()
        np.random.seed(seed=2)  # prob = 0.4
        # self.assertEqual(d.check_if_aggravate(np.arange(10)).all(), False)
        # self.assertEqual(d.check_if_aggravate(np.arange(10), s_i=0.1).all(), True)

    def test_append_row_to_df(self):
        d = sim.Data()
        self.assertEqual(len(d.df), 0)
        sample_df = pd.DataFrame(
            [range(len(d.df.columns))], columns=d.df.columns, index=["sample_id"]
        )
        d.append(sample_df)
        self.assertEqual(len(d.df), 1)
        with self.assertRaises(ValueError):
            d.append(sample_df)


class ContangionViaSQL(unittest.TestCase):
    # Tests that require connection to the server in Milan

    @unittest.skipIf(SKIP_TESTS, "Skip SQL tests")
    def test_sql_query_contagion(self):
        default_config = sim.test_conf
        set_of_mock_potential_patients = {
            ".QP/64EdoTcdkMnmXGVO0A",
            "BP51jL2myIMRqfYseLbGfM,D8hZWX/ycJMmF4qg1uGkZc",
            " FNGiD7T4cpkOIM3mq.YdMY",
            "HhulO23UWA2BVHqsECvjJY",
            "Sq1s6KEGp1Qm8MN1o1paM.",
            "cMvEW1y.DLUsMgtP951/f.",
            "czqASEiMg7MUBvYLidDHZY",
            "m5YJXcVamIkxaZmrDw1mwA",
            "xDK0mIGasmAilJrvnFS3Pw",
        }
        self.assertEqual(
            len(
                sim.contagion_in_sql(
                    set_of_mock_potential_patients, default_config, "2012-03-29"
                )
            ),
            7942,
        )


class PickleToSet(unittest.TestCase):
    def test_pickle_to_set(self):
        self.assertTrue(
            type(sim.one_array_pickle_to_set("data/destination_ids_first_3days.pickle"))
            is set
        )


class Infection(unittest.TestCase):
    # def test_check_if_infected(self):
    #     self.assertTrue(sim.Data.check_if_infected(1, 1))
    #     self.assertTrue(sim.Data.check_if_infected(0.5, 0.5))
    #     self.assertTrue(sim.Data.check_if_infected(0.1, 1) is True)
    #     self.assertTrue(sim.Data.check_if_infected(0.1, 1, 0.05))

    def test_infection_state_transition_return_values(self):
        d = sim.Data()
        infected = pd.DataFrame(
            [[50]], columns=["daily_duration"], index=[".QP/64EdoTcdkMnmXGVO0A"]
        )
        df = d.infection_state_transition(infected, date(2012, 3, 26))
        self.assertEqual(
            set(df["age_group"]) - set(range(len(sim.test_conf["age_dist"]))), set()
        )
        self.assertTrue(df["color"].isin(np.array([True, False])).all())
        self.assertTrue((df["expiration_date"].values > date(2012, 3, 26)).all())

    def test_is_enough_duration(self):
        d = sim.Data()
        df_50 = pd.DataFrame([50], columns=["daily_duration"])
        df_10 = pd.DataFrame([10], columns=["daily_duration"])
        self.assertEqual(
            d.is_enough_duration(df_50["daily_duration"]), np.array([True])
        )
        self.assertEqual(
            d.is_enough_duration(df_10["daily_duration"]), np.array([False])
        )

    def test_get_trajectory(self):
        d = sim.Data()
        curr_date = date(2012, 3, 26)
        infected = sim.contagion_in_csv(
            "./data/mock_data.csv", {"FNGiD7T4cpkOIM3mq.YdMY"}, curr_date
        )
        sim.get_trajectory(infected, d, curr_date, add_duration=False)
        self.assertListEqual(
            d.df.columns.tolist(),
            [
                "age_group",
                "color",
                "infection_date",
                "expiration_date",
                "daily_duration",
            ],
        )
        self.assertEqual(d.df.index.name, "id")


class GoogleCloudTest(unittest.TestCase):
    # TODO: add mock
    def __init__(self, *args, **kwargs):
        super(GoogleCloudTest, self).__init__(*args, **kwargs)
        conf = sim.BasicConfiguration()
        self.gcloud = sim.GoogleCloud(conf)

    def test_csv_validity(self):
        pass

    def test_connection_to_google_cloud_storage(self):
        bucket = self.gcloud.s_client.lookup_bucket("simulation_runs")
        self.assertIsInstance(bucket, storage.Bucket)

    @unittest.skipIf(SKIP_TESTS, "Skip Google Storage Tests")
    def test_download(self):
        self.gcloud.download("test.csv")
        self.assertTrue(Path("./data/test.csv").exists())

    @unittest.skipIf(SKIP_TESTS, "Skip Google Storage Tests")
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
        self.assertNotIn(
            False, [isinstance(t, datastore.entity.Entity) for t in all_tasks]
        )

    def test_add_task(self):
        self.gcloud.ds_client.put = MagicMock(return_value=True)
        self.assertTrue(
            self.gcloud.add_task({"dataset": "h3g"}, sim.TaskConfig(sim.test_conf))
        )


if __name__ == "__main__":
    unittest.main()
