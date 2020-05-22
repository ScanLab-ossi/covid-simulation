import psycopg2
from random import seed, randint, choices, sample
import multiprocessing as mp
import numpy as np

from simulation.helpers import timing
from simulation.dataset import Dataset
from simulation.task_config import TaskConfig


class CSVContagion(object):
    def __init__(self, dataset: Dataset, task_conf: TaskConfig):
        self.dataset = dataset
        self.task_conf = task_conf

    def pick_patient_zero(
        self, set_of_potential_patients=None, arbitrary_patient_zero: list = [],
    ):
        if arbitrary_patient_zero:
            return set(arbitrary_patient_zero)
        elif set_of_potential_patients:
            seed(1)
            randomly_patient_zero = sample(
                set_of_potential_patients, self.task_conf.get("number_of_patient_zero")
            )
            return set(randomly_patient_zero)
        else:
            first_day = self.dataset.data[
                self.dataset.data["datetime"].dt.date == self.dataset.start_date
            ]
            return set(
                first_day[["source", "destination"]]
                .stack()
                .drop_duplicates()
                .reset_index(drop=True)
                .sample(self.task_conf.get("number_of_patient_zero"))
            )

    @timing
    def contagion(self, infected_set, curr_date=None):
        df = self.dataset.data
        contagion_df = (
            df[
                (df["source"].isin(infected_set) | df["destination"].isin(infected_set))
                & (df["datetime"].dt.date == curr_date)
            ]
            .melt(id_vars=["datetime", self.dataset.infection_param])
            .drop(columns=["variable"])
            .drop_duplicates()
            .groupby("value")[self.dataset.infection_param]
            .sum()
            .to_frame(name="daily_duration")
            .reset_index()
            .rename(columns={"value": "id"})
        )
        contagion_df = contagion_df[~contagion_df["id"].isin(infected_set)].set_index(
            "id"
        )
        return contagion_df


class SQLContagion(object):
    def __init__(self, dataset: Dataset, task_conf: TaskConfig):
        self.dataset = dataset
        self.task_conf = task_conf

    @timing
    def pick_patient_zero(
        self, set_of_potential_patients=None, arbitrary_patient_zero: list = [],
    ):
        # return set of zero patients
        if arbitrary_patient_zero:
            return set(arbitrary_patient_zero)
        else:
            seed(1)
            randomly_patient_zero = sample(
                set_of_potential_patients, self.task_conf.get("number_of_patient_zero")
            )
            return set(randomly_patient_zero)

    @timing
    def query_sql_server(self, subset_of_infected_set, date, basic_conf):
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
    def contagion(self, infected_set, basic_conf, date):
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
