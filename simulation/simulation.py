from random import seed, randint, choices
import random, pickle, os, configparser, json
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import psycopg2
from google.cloud import storage, datastore
import requests
from pathlib import Path
from jsonschema import Draft7Validator, validators
from typing import Union
from collections import UserDict

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


class BasicConfiguration(object):
    """
    Initialize all *technical* configuration aspects (postgres, google)
    This is not the configuration for the simulation run (called that task_conf)
    """

    def __init__(self, config_file: Path = Path("secrets.conf")):
        self.config_file = config_file
        self.config = self.get_config()
        self.set_env()

    def get_config(self):
        conf = configparser.ConfigParser()
        conf.read(self.config_file)
        return conf

    def set_env(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.config["google"][
            "credentials"
        ]


class TaskConfig(UserDict):
    """
    Config for each simulation run. Can be used for validation
    """

    def __init__(self, data, validate=True):
        super().__init__(dict(data))
        with open(Path("./data/config_schema.json")) as f:
            schema = json.load(f)
        self.validator = self.create_validator(schema)
        if validate:
            self.validator.validate(self.data)
            self.is_valid = self.validator.is_valid(self.data)
        self.data = self.as_ndarrays()

    def create_validator(self, schema):
        type_checker = Draft7Validator.TYPE_CHECKER.redefine(
            "array",
            fn=lambda checker, instance: True
            if isinstance(instance, np.ndarray) or isinstance(instance, list)
            else False,
        )
        Validator = validators.extend(Draft7Validator, type_checker=type_checker)
        return Validator(schema=schema)

    def as_lists(self):
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.data.items()
        }

    def as_ndarrays(self):
        return {
            k: (np.array(v) if isinstance(v, list) else v)
            for k, v in dict(self.data).items()
        }


class Data(object):
    def __init__(self, output_filename="output"):
        self.df = pd.DataFrame(
            columns=["id", "age_group", "color", "infection_date", "expiration_date"]
        )
        self.output_path = Path(f"./data/{output_filename}.csv")

    def display(self):
        print(self.df.to_string())

    def shape(self):
        print(self.df.shape)

    def export(self):
        self.df.to_csv(self.output_path, index=False)

    def append(self, id, age_group, infection_date, expiration_date, color):
        self.df = self.df.append(
            {
                "id": id,
                "age_group": age_group,
                "infection_date": infection_date,
                "expiration_date": expiration_date,
                "color": color,
            },
            ignore_index=True,
        )

    @staticmethod
    def time_to_recovery(infection_date, task_conf):
        mu, sigma = (
            task_conf.get("recovery_time_dist")[0],
            task_conf.get("recovery_time_dist")[1],
        )
        recovery_duration = int(np.around(np.random.normal(mu, sigma)))
        if recovery_duration <= 1:  # Avoid the paradox of negative recovery duration.
            recovery_duration = 1
        expiration_date = datetime.strptime(infection_date, "%Y-%m-%d") + timedelta(
            recovery_duration
        )
        return expiration_date.strftime("%Y-%m-%d")

    @staticmethod
    def time_to_aggravation(infection_date, task_conf):
        mu, sigma = (
            task_conf.get("aggravation_time_dist")[0],
            task_conf.get("aggravation_time_dist")[1],
        )
        aggravation_duration = int(np.around(np.random.normal(mu, sigma)))
        if (
            aggravation_duration <= 1
        ):  # Avoid the paradox of negative recovery duration.
            aggravation_duration = 1
        expiration_date = datetime.strptime(infection_date, "%Y-%m-%d") + timedelta(
            aggravation_duration
        )
        return expiration_date.strftime("%Y-%m-%d")

    @staticmethod
    def check_if_aggravate(age_group=None, s_i=0.7):
        # TO BE MORE COMPETABILE TO THE MODEL
        threshold = s_i
        prob = np.random.rand(1)
        return prob > threshold

    @staticmethod
    def check_if_infected(P_max, P_gb, threshold=0.05):
        return P_max * P_gb > threshold

    @staticmethod
    def infection_state_transition(task_conf, infection_date):
        # get info about an infected person and return relevant data for dataframe
        age_dist = task_conf.get("age_dist")
        age_group = choices(np.arange(len(age_dist)), age_dist)
        # print(age_dist, np.arange(len(age_dist)), age_group)
        if Data.check_if_aggravate(age_group=age_group):
            color = "purple"
            expiration_date = Data.time_to_aggravation(infection_date, task_conf)
        else:
            color = "blue"
            expiration_date = Data.time_to_recovery(infection_date, task_conf)
        return age_group, color, expiration_date


class GoogleCloud(object):
    def __init__(
        self, config: BasicConfiguration, bucket_name: str = "simulation_runs"
    ):
        self.config = config
        self.s_client = storage.Client()
        self.bucket = self.s_client.bucket(bucket_name)
        self.ds_client = datastore.Client()
        self.todo = []
        self.done = []
        self.task_validator = Draft7Validator(
            {
                "type": "object",
                "properties": {"dataset": {"type": "string", "enum": ["h3g"]}},
                "required": ["dataset"],
            }
        )

    def upload(self, filename: Path, new_name: str):
        then = datetime.now()
        blob = self.bucket.blob(f"{new_name}.csv")
        with open(filename, "rb") as f:
            if file_size := os.path.getsize(filename) < 10_485_760:  # 10MB
                blob.upload_from_file(f, content_type="text/csv")
            else:
                url = blob.create_resumable_upload_session(
                    content_type="text/csv", size=file_size
                )
                res = requests.put(url, data=f)
                res.raise_for_status()
        print(
            f"uploaded {new_name} to {self.bucket.name}. took {datetime.now() - then}"
        )
        return blob.self_link

    def get_tasklist(self, done=False):
        query = self.ds_client.query(kind="task")
        result = list(query.fetch())  # .add_filter("done", "=", done)
        self.done = [t for t in result if t["done"] == True]
        self.todo = [t for t in result if t["done"] == False]

    def add_task(self, task_dict: dict, config_dict: TaskConfig):
        self.task_validator.validate(task_dict)
        task_key = self.ds_client.key("task")  # , np.random.randint(1e15, 1e16))
        task = datastore.Entity(key=task_key)
        task.update(
            {
                **task_dict,
                **{
                    "done": False,
                    "task_added": datetime.now(),
                    "config": config_dict.as_lists(),
                },
            }
        )
        return self.ds_client.put(task)
        # print("added task!")

    def write_results(self, result):
        if len(result) > 1:
            batch = self.ds_client.batch()
            with batch:
                tasks = []
                for data, task in result:
                    link = self.upload(data.output_filename, new_name=task.id)
                    task.update(
                        {"done": True, "task_done": datetime.now(), "link": link}
                    )
                    tasks.append(task)
                self.ds_client.put_multi(tasks)
        else:
            data, task = result[0]
            link = self.upload(data.output_filename, new_name=task.id)
            task.update({"done": True, "task_done": datetime.now(), "link": link})
            self.ds_client.put(task)


def pick_patient_zero(
    set_of_potential_patients,
    num_of_patients=1,
    random_seed=1,
    arbitrary=False,
    arbitrary_patient_zero=["MJviZSTPuYw1v0W0cURthY"],
):
    # return set of zero patients
    if arbitrary:
        return arbitrary_patient_zero
    else:
        seed(random_seed)
        randomly_patient_zero = random.sample(
            set_of_potential_patients, num_of_patients
        )
        return randomly_patient_zero


def get_active_ids(
    data,
):  # return all the people who make contact in the given dataset (data is csv or sql query)
    active_ids = set()
    with open(data) as fp:
        for line in fp:
            active_ids.add(line.split(",")[0])
            active_ids.add(line.split(",")[5])
    return active_ids


def contagion_in_csv(data_as_csv_file, infected_list, date):
    contagion_set = set()
    with open(data_as_csv_file) as fp:
        for line in fp:
            if line.split(",")[2] == date:
                if line.split(",")[0] in infected_list:
                    contagion_set.add(line.split(",")[5])
                if line.split(",")[5] in infected_list:
                    contagion_set.add(line.split(",")[0])

    return contagion_set


def contagion_in_sql(infected_set, basic_conf, date):
    if len(infected_set) == 0:  # empty set
        return set()
    # data format is "YYYY-MM-DD"
    conn = psycopg2.connect(**basic_conf.config["postgres"])
    # Open a cursor to perform database operations
    cur = conn.cursor()
    infected_tuple = str(tuple(infected_set))
    date_as_str = "'" + str(date) + "'"
    # Query the database and obtain data as Python objects
    query = f"""select  destination as a
                   from h3g.call
                   where date between {date_as_str} and {date_as_str}
                   AND
                   source in {infected_tuple}
                   Union
                   select source as a
                   from h3g.call
                   where date between {date_as_str} and {date_as_str}
                   AND
                   destination in {infected_tuple};"""
    print(len(query))
    cur.execute(query)
    cur_out_as_arr = np.asarray(cur.fetchall())
    contagion_set = set(cur_out_as_arr.flatten())
    # Close communication with the database
    cur.close()
    conn.close()
    return contagion_set


def daily_duration_in_sql(id, date):
    # to be completed
    return 50


def virus_spread(data, set_of_patients, start_date, days):
    date_str = start_date
    date_object = datetime.strptime(date_str, "%Y-%m-%d")
    patients = set_of_patients
    for i in range(days):
        new_patients = contagion_in_csv(
            data, patients, date_object.strftime("%Y-%m-%d")
        )
        patients = patients.union(new_patients)
        date_object += timedelta(days=1)
    return patients


def one_array_pickle_to_set(pickle_file_name):
    # open a file, where you stored the pickled data
    with open(pickle_file_name, "rb") as f:
        data = pickle.load(f)
    set_from_arr = set(data.flatten())
    return set_from_arr


def contagion_runner(data, basic_conf, task_conf):
    period = 10  # max is 65
    first_date = date(2012, 3, 26)  #
    pickle_of_ids = one_array_pickle_to_set(
        Path("./data/destination_ids_first_3days.pickle")
    )
    zero_patients = pick_patient_zero(
        pickle_of_ids, num_of_patients=task_conf.get("number_of_patient_zero")
    )
    # -- TO DO -- append the zero patients into dataframe
    for id in zero_patients:
        age_group, color, expiration_date = data.infection_state_transition(
            task_conf, first_date.strftime("%Y-%m-%d")
        )
        data.append(
            id, age_group, first_date.strftime("%Y-%m-%d"), expiration_date, color
        )

    set_of_infected = set(zero_patients)
    for day in range(period):
        curr_date = first_date + timedelta(days=day)
        curr_date_as_str = curr_date.strftime("%Y-%m-%d")
        set_of_contact_with_patient = contagion_in_sql(
            set_of_infected, basic_conf, curr_date_as_str
        )
        for id in set_of_contact_with_patient:
            daily_duration = daily_duration_in_sql(id, curr_date_as_str)  # = D_i
            P_gb = (
                daily_duration / task_conf.get("D_max")
                if daily_duration > task_conf.get("D_min")
                else 0
            )
            P_max = task_conf.get("P_max")
            if data.check_if_infected(P_max, P_gb):
                age_group, color, expiration_date = data.infection_state_transition(
                    task_conf, curr_date_as_str
                )
                data.append(id, age_group, curr_date_as_str, expiration_date, color)

        set_of_infected = set(
            data.df[(data.df["expiration_date"] > curr_date_as_str)]["id"].values
        )
        # patients that doesn't recovered or died yet

        print(f"Status of {day}:")
        data.display()

    data.export()


def main(test_conf: dict = False):
    basic_conf = BasicConfiguration()
    gcloud = GoogleCloud(basic_conf)
    if test_conf:
        contagion_runner(Data(), basic_conf=basic_conf, task_conf=TaskConfig(test_conf))
    else:
        gcloud.get_tasklist()
        if gcloud.todo:
            results = []
            for task in gcloud.todo:
                data = Data(output_path=task.id)
                contagion_runner(
                    data, basic_conf=basic_conf, task_conf=TaskConfig(task)
                )
                results.append((data, task))
            gcloud.write_results(results)
        else:
            print("No tasks waiting!")


if __name__ == "__main__":
    main(test_conf)
