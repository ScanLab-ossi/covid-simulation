import unittest
from unittest import mock
from unittest.mock import MagicMock
import numpy as np
from random import seed
from datetime import datetime
import pandas as pd
import os, configparser
from google.cloud import storage
from google.cloud import datastore
from pathlib import Path

import simulation.simulation as sim


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
        self.assertEqual(
            sim.contagion_in_csv(
                "./data/mock_data.csv", {"FNGiD7T4cpkOIM3mq.YdMY"}, "2012-03-26"
            ),
            {"ODOkY9pchzsDHj.23UGQoc"},
        )

    def test_first_circle_of_2_patient_in_specific_date(self):
        self.assertEqual(
            sim.contagion_in_csv(
                "./data/mock_data.csv",
                {"..7cyMMPqV.bMVjsN7Rcns", "..cvdr3nnY2eZmwko9evCQ"},
                "2012-03-26",
            ),
            {"cMvEW1y.DLUsMgtP951/f."},
        )

    def test_virus_spread_over_some_days(self):
        self.assertEqual(
            len(
                sim.virus_spread(
                    "./data/mock_data.csv",
                    {
                        "FNGiD7T4cpkOIM3mq.YdMY",
                        "Sq1s6KEGp1Qm8MN1o1paM.",
                        "HhulO23UWA2BVHqsECvjJY",
                    },
                    "2012-04-02",
                    2,
                )
            ),
            12,
        )


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


class DataSturcture(unittest.TestCase):
    def test_create_empty_dataframe(self):
        d = sim.Data()
        # d.display()

    def test_time_to_recovery(self):
        default_config = sim.test_conf
        np.random.seed(seed=1)
        self.assertEqual(
            sim.Data.time_to_recovery("2012-04-02", default_config), "2012-05-08"
        )

    def test_time_to_aggravation(self):
        default_config = sim.test_conf
        np.random.seed(seed=2)
        self.assertEqual(
            sim.Data.time_to_aggravation("2012-04-02", default_config), "2012-04-06"
        )

    def test_check_if_aggravation(self):
        default_config = sim.test_conf
        np.random.seed(seed=2)  # prob = 0.4
        self.assertEqual(sim.Data.check_if_aggravate(), False)
        np.random.seed(seed=2)  # prob = 0.4
        self.assertEqual(sim.Data.check_if_aggravate(s_i=0.1), True)

    def test_append_row_to_df(self):
        d = sim.Data()
        default_config = sim.test_conf
        for i in range(15):
            age_group, color, expiration_date = d.infection_state_transition(
                default_config, "2012-03-29"
            )
            d.append(
                id="a" + str(i),
                age_group=age_group,
                infection_date="2012-03-29",
                expiration_date=expiration_date,
                color=color,
            )
        d.display()

        print(set(d.df[(d.df["expiration_date"] > "2012-04-16")]["id"].values))


class ContangionViaSQL(unittest.TestCase):
    skip_sql_tests = (
        True  # There is tests that required connection to the server in Milan
    )

    @unittest.skipIf(skip_sql_tests, "Skip SQL tests")
    def test_sql_query_contagion(self):
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
            len(sim.contagion_in_sql(set_of_mock_potential_patients, "2012-03-29")),
            7942,
        )


class PickleToSet(unittest.TestCase):
    def test_pickle_to_set(self):
        self.assertTrue(
            type(sim.one_array_pickle_to_set("data/destination_ids_first_3days.pickle"))
            is set
        )


class Infection(unittest.TestCase):
    def test_check_if_infected(self):
        self.assertTrue(sim.Data.check_if_infected(1, 1))
        self.assertTrue(sim.Data.check_if_infected(0.5, 0.5))
        self.assertTrue(sim.Data.check_if_infected(0.1, 1) is True)
        self.assertTrue(sim.Data.check_if_infected(0.1, 1, 0.05))

    def test_infection_state_transition_age_group(self):
        default_config = sim.test_conf
        self.assertTrue(
            sim.Data.infection_state_transition(default_config, "2012-03-29")[0]
            in np.arange(len(default_config.get("age_dist")))
        )

    def test_infection_state_transition_return_values(self):
        default_config = sim.test_conf
        self.assertTrue(
            sim.Data.infection_state_transition(default_config, "2012-03-29")[0]
            in np.arange(len(default_config.get("age_dist")))
        )
        self.assertTrue(
            sim.Data.infection_state_transition(default_config, "2012-03-29")[1]
            in ["blue", "purple"]
        )
        expiration_date = datetime.strptime(
            sim.Data.infection_state_transition(default_config, "2012-03-29")[2],
            "%Y-%m-%d",
        )
        self.assertTrue(expiration_date > datetime.strptime("2012-03-29", "%Y-%m-%d"))


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

    def test_upload(self):
        pass

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
