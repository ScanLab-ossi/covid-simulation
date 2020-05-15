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
from datetime import datetime
import multiprocessing as mp

try:
    from helpers import timing
except ModuleNotFoundError:
    from simulation.helpers import timing

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

    def __init__(self):
        self.add_keyfile()
        self.config = self.get_config()

    def get_config(self):
        conf_file = Path("secrets.conf")
        conf = configparser.ConfigParser()
        if not conf_file.exists():
            conf["postgres"] = {
                "dbname": os.environ["POSTGRES_DATABASE_NAME"],
                "user": os.environ["POSTGRES_USER_NAME"],
                "password": os.environ["POSTGRES_USER_PASSWORD"],
                "host": os.environ["POSTGRES_DATABASE_HOST"],
            }
        else:
            conf.read(conf_file)
        return conf

    def add_keyfile(self):
        if not Path("./keyfile.json").exists():
            with open("keyfile.json", "w") as fp:
                json.dump(json.loads(os.environ["KEYFILE"]), fp)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"


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
    def __init__(self, output_filename="output", task_conf=test_conf):
        self.df = pd.DataFrame(
            columns=["age_group", "color", "infection_date", "expiration_date"]
        )
        self.df.index.name = "id"
        self.output_path = Path(f"./data/{output_filename}.csv")
        self.task_conf = task_conf
        self.first_date = date(2012, 3, 26)

    def display(self):
        print(self.df.to_string())

    def shape(self):
        print(self.df.shape)

    def export(self):
        self.df.to_csv(self.output_path)

    def append(self, new_df):
        self.df = self.df.append(new_df, verify_integrity=True)

    def expiration_date(self, color, infection_date):
        time_dist = "recovery_time_dist" if color else "aggravation_time_dist"
        duration = int(np.around(np.random.normal(*self.task_conf.get(time_dist))))
        if duration <= 1:  # Avoid the paradox of negative recovery duration.
            duration = 1
        expiration_date = infection_date + timedelta(duration)
        return expiration_date

    @timing
    def check_if_aggravate(self, age_group, s_i=0.7):
        # TO BE MORE COMPETABILE TO THE MODEL
        return np.random.rand(len(age_group)) > s_i

    @timing
    def is_enough_duration(self, daily_duration):
        return (
            np.where(
                daily_duration.values >= self.task_conf.get("D_min"),
                daily_duration.values / self.task_conf.get("D_max"),
                0,
            )
            * self.task_conf.get("P_max")
            > 0.05
        )

    @timing
    def infection_state_transition(self, infected, infection_date):
        # get info about an infected person and return relevant data for dataframe
        df = pd.DataFrame(
            np.random.choice(
                len(self.task_conf.get("age_dist")),
                len(infected.index),
                p=self.task_conf.get("age_dist").tolist(),
            ),
            columns=["age_group"],
            index=infected.index,
        )
        df["color"] = self.check_if_aggravate(df["age_group"].values)
        df["expiration_date"] = df["color"].apply(
            self.expiration_date, args=(infection_date,)
        )
        return df


class GoogleCloud(object):
    def __init__(self, config: BasicConfiguration):
        self.config = config
        self.s_client = storage.Client()
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

    def upload(
        self, filename: Path, new_name: str = None, bucket_name: str = "simulation_runs"
    ):
        then = datetime.now()
        bucket = self.s_client.bucket(bucket_name)
        blob_name = f"{new_name}.csv" if new_name else filename.name
        blob = bucket.blob(blob_name)
        with open(filename, "rb") as f:
            file_size = os.path.getsize(filename)
            if file_size < 10_485_760:  # 10MB
                blob.upload_from_file(f, content_type="text/csv")
            else:
                url = blob.create_resumable_upload_session(
                    content_type="text/csv", size=file_size
                )
                res = requests.put(url, data=f)
                res.raise_for_status()
        print(f"uploaded {blob_name} to {bucket.name}. took {datetime.now() - then}")
        return blob.self_link

    def download(self, blob_name: str):
        destination_path = Path(f"./data/{blob_name}")
        if not destination_path.exists():
            bucket = self.s_client.bucket("simulation_datasets")
            blob = bucket.blob(blob_name)
            print(f"downloading {blob_name}. this might take a while.")
            blob.download_to_filename(destination_path)
            print(f"finished downloading {blob_name}. thank you for your patience :)")
        else:
            print(f"{destination_path.name} already exists")

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


@timing
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


@timing
def get_active_ids(
    data,
):  # return all the people who make contact in the given dataset (data is csv or sql query)
    active_ids = set()
    with open(data) as fp:
        for line in fp:
            active_ids.add(line.split(",")[0])
            active_ids.add(line.split(",")[5])
    return active_ids


@timing
def contagion_in_csv(data_path, infected_set, date=None):
    df = pd.read_csv(data_path, header=None, parse_dates=[2])
    contagion_df = (
        df[
            (df[0].isin(infected_set) | df[5].isin(infected_set))
            & (df[2] == np.datetime64(date))
        ][[0, 1, 2, 3, 5]]
        .melt(id_vars=[1, 2, 3])
        .drop(columns=["variable"])
        .drop_duplicates()
        .groupby("value")[1]
        .sum()
        .to_frame(name="daily_duration")
        .reset_index()
        .rename(columns={"value": "id"})
    )
    contagion_df = contagion_df[~contagion_df["id"].isin(infected_set)].set_index("id")
    return contagion_df


@timing
def query_sql_server(subset_of_infected_set, date, basic_conf):
    # data format is "YYYY-MM-DD"
    conn = psycopg2.connect(**basic_conf.config["postgres"])
    # Open a cursor to perform database operations
    cur = conn.cursor()
    infected_tuple = str(tuple(subset_of_infected_set))
    infected_tuple_in_postrgers_format = (
        infected_tuple.replace("(", "{").replace(")", "}").replace("'", "")
    )
    # for the next query the postgres is awaiting to {a, b, c, d} such that a,...,d are id's.
    date_as_str = "'" + str(date) + "'"
    # Query the database and obtain data as Python objects
    query = f"""select  distinct destination as a
                   from h3g.call
                   where date between {date_as_str} and {date_as_str}
                   AND
                   source = any('{infected_tuple_in_postrgers_format}'::text[])
                    AND
                   destination != any('{infected_tuple_in_postrgers_format}'::text[])
                   Union
                   select distinct source as a
                   from h3g.call
                   where date between {date_as_str} and {date_as_str}
                   AND
                   destination = any('{infected_tuple_in_postrgers_format}'::text[])
                    AND
                   source != any('{infected_tuple_in_postrgers_format}'::text[]);"""
    # print(f"Query length is {len(query)}, over 1GB could be problem")
    cur.execute(query)
    cur_out_as_arr = np.asarray(cur.fetchall()).flatten()
    # Close communication with the database
    contagion_list = list(cur_out_as_arr)
    cur.close()
    conn.close()
    return contagion_list


@timing
def contagion_in_sql(infected_set, basic_conf, date):
    if len(infected_set) == 0:  # empty set
        return set()
    # data format is "YYYY-MM-DD"

    infected_list = list(infected_set)
    batch_size = 2000
    sub_infected_list = [
        infected_list[x : x + batch_size]
        for x in range(0, len(infected_list), batch_size)
    ]

    pool = mp.Pool(processes=4)
    results = [
        pool.apply(query_sql_server, args=(sub_infected_list[x], date, basic_conf))
        for x in range(len(sub_infected_list))
    ]

    flat_results = [item for sublist in results for item in sublist]
    return set(flat_results)


@timing
def daily_duration_in_sql(id=None, date=None):
    # to be completed
    return 90


@timing
def virus_spread(data_path, set_of_patients, start_date, days):
    patients = pd.DataFrame(index=set_of_patients)
    for i in range(days):
        new_patients = contagion_in_csv(data_path, patients.index.tolist(), start_date)
        patients = patients.append(new_patients, verify_integrity=True)
        start_date += timedelta(days=1)
    return patients


@timing
def one_array_pickle_to_set(pickle_file_name):
    # open a file, where you stored the pickled data
    with open(pickle_file_name, "rb") as f:
        data = pickle.load(f)
    set_from_arr = set(data.flatten())
    return set_from_arr


@timing
def get_trajectory(infected, data, curr_date, add_duration=True):
    if isinstance(infected, set):
        infected = pd.DataFrame(index=infected)
        infected.index.name = "id"
        if add_duration:
            infected["daily_duration"] = daily_duration_in_sql()  # = D_i
    infected = infected.loc[
        ~infected.index.isin(data.df.index)
    ]  # remove newly infected if they've already been infected in the past
    if "daily_duration" in infected.columns:
        infected = infected[data.is_enough_duration(infected["daily_duration"])]
    infected = infected.join(data.infection_state_transition(infected, curr_date))
    infected["infection_date"] = curr_date
    data.append(infected)


@timing
def contagion_runner(data, basic_conf, sql=True):
    period = 10  # max is 65
    if sql:
        pickle_of_ids = one_array_pickle_to_set(
            Path("./data/destination_ids_first_3days.pickle")
        )
        zero_patients = pick_patient_zero(
            pickle_of_ids, num_of_patients=data.task_conf.get("number_of_patient_zero")
        )
    else:
        zero_patients = set(
            pd.read_csv(
                "../call_1.csv",
                header=None,
                nrows=data.task_conf.get("number_of_patient_zero"),
            )[0]
        )
    get_trajectory(zero_patients, data, data.first_date, add_duration=False)
    set_of_infected = zero_patients
    for day in range(period):
        print(f"Status of {day}:")
        curr_date = data.first_date + timedelta(days=day)
        if sql:
            set_of_contact_with_patient = contagion_in_sql(
                set_of_infected, basic_conf, curr_date.strftime("%Y-%m-%d")
            )
        else:
            set_of_contact_with_patient = contagion_in_csv(
                "../call_1.csv", set_of_infected, date=curr_date
            )
        start_time = datetime.now()
        get_trajectory(set_of_contact_with_patient, data, curr_date)
        set_of_infected = set(data.df[data.df["expiration_date"] > curr_date].index)
        # patients that haven't recovered or died yet
        print(f"local calculation time: {(datetime.now() - start_time)}")
        print(data.shape())

    data.export()


def main(test_conf: dict = False):
    basic_conf = BasicConfiguration()
    gcloud = GoogleCloud(basic_conf)
    if test_conf:
        contagion_runner(
            Data(task_conf=test_conf), basic_conf=basic_conf
        )  # , sql=False)
    else:
        gcloud.get_tasklist()
        if gcloud.todo:
            results = []
            for task in gcloud.todo:
                data = Data(output_path=task.id, task_conf=TaskConfig(task))
                contagion_runner(data, basic_conf=basic_conf)
                results.append((data, task))
            gcloud.write_results(results)
        else:
            print("No tasks waiting!")


if __name__ == "__main__":
    main(test_conf)
