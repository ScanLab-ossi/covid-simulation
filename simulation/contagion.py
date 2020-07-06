try:
    import psycopg2
except ModuleNotFoundError:
    pass
from random import seed, randint, choices, sample
import multiprocessing as mp
import numpy as np
import numpy.ma as ma
import pandas as pd
from datetime import date, datetime, timedelta
import os, logging
from typing import Union

from simulation.helpers import timing, one_array_pickle_to_set
from simulation.state_transition import StateTransition
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.output import Output
from simulation.constants import *


class ContagionRunner(object):
    """Runs one batch"""

    @staticmethod
    def contagion_runner(
        dataset: Dataset, output: Output, task: Task, reproducable: bool = False,
    ) -> Output:
        contagion = (
            CSVContagion(dataset, task)
            if dataset.storage == "csv"
            else SQLContagion(dataset, task)
        )
        for i in range(task["ITERATIONS"]):
            start = datetime.now()
            if not settings["PARALLEL"]:
                print(f"repetition {i}")
            st = StateTransition(dataset, task)
            if dataset.storage == "sql":
                pickle_of_ids = one_array_pickle_to_set(
                    Path(DATA_FOLDER / "destination_ids_first_3days.pickle")
                )
                zero_patients = contagion.pick_patient_zero(
                    pickle_of_ids, num_of_patients=task["number_of_patient_zero"],
                )
            else:
                zero_patients = contagion.pick_patient_zero(reproducable=reproducable)
            zero_patients = st.get_trajectory(
                zero_patients, output, dataset.start_date,
            )
            output.append(zero_patients)
            active_infected = output.df[["color"]]
            for day in range(dataset.period + 1):
                if settings["VERBOSE"]:
                    process = f", process {os.getpid()}" if settings["PARALLEL"] else ""
                    print(f"Status of {day}:" + process)
                curr_date = dataset.start_date + timedelta(days=day)
                newly_infected = contagion.contagion(
                    active_infected, curr_date=curr_date
                )
                if len(newly_infected) != 0:
                    newly_infected = st.get_trajectory(
                        newly_infected, output, curr_date
                    )
                    output.append(newly_infected)
                active_infected = output.df[output.df["transition_date"] > curr_date][
                    ["color"]
                ]
                # patients that haven't recovered or died yet
                if settings["VERBOSE"]:
                    print(f"{output.df.shape[0]} infected today altogether")
            output.batch.append(output.df)
            # output.export(filename=(str(task.id)), how="df", pickle=True)
            output.reset()
            print(f"repetition {i} took {datetime.now() - start}")
            logging.info(f"repetition {i} took {datetime.now() - start}")
        return output


class Contagion(object):
    def __init__(self, dataset: Dataset, task: Task):
        self.dataset = dataset
        self.task = task
        self.rng = np.random.default_rng()

    def _cases(self, df: pd.DataFrame, D_i: str = "duration") -> pd.DataFrame:
        if self.task["infection_model"] == 1:
            df[D_i] = np.where(
                df[D_i].values >= self.task["D_min"],
                np.minimum(
                    df[D_i].values / self.task["D_max"] * self.task["P_max"], 1.0
                ),
                0,
            )
        elif self.task["infection_model"] == 2:
            hops = df["hops"].values if self.dataset.hops else 1
            df[D_i] = np.where(
                df[D_i].values >= self.task["D_min"], df[D_i].values / hops, 0.00001,
            )
        return df

    def _is_infected(self, P_gi_i: pd.Series) -> np.ndarray:
        return np.vectorize(lambda x: self.rng.choice([True, False], 1, p=[x, 1 - x]))(
            P_gi_i.values
        )

    def _multiply_not_infected_chances(self, d_i_k: pd.Series) -> float:
        return 1 - np.prod(
            1 - np.minimum(d_i_k.values / self.task["D_max"] * self.task["P_max"], 1)
        )

    def _consider_alpha(self, contagion_df: pd.DataFrame) -> pd.DataFrame:
        new_duration = (
            ma.array(
                contagion_df[self.dataset.infection_param].values,
                mask=contagion_df["color"].values,
            )
            * (1 - self.task["alpha_blue"])
        ).data
        new_duration[new_duration > self.task["D_max"]] = self.task["D_max"]
        contagion_df["duration"] = new_duration
        return contagion_df


class CSVContagion(Contagion):
    def pick_patient_zero(
        self,
        set_of_potential_patients: Union[set, None] = None,
        arbitrary_patient_zero: list = [],
        reproducable: bool = False,
    ) -> set:
        if arbitrary_patient_zero:
            return set(arbitrary_patient_zero)
        elif set_of_potential_patients:
            # seed_ = os.getpid() if PARALLEL else 1
            # seed(seed_)
            randomly_patient_zero = np.random.choice(
                list(set_of_potential_patients),
                self.task["number_of_patient_zero"],
                replace=False,
            )

            # randomly_patient_zero = sample(
            #     set_of_potential_patients, self.task["number_of_patient_zero"]
            # )
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
                .sample(
                    self.task["number_of_patient_zero"],
                    random_state=(42 if reproducable else None),
                )
            )

    @timing
    def contagion(
        self, infected: pd.DataFrame, curr_date: date = None,
    ) -> pd.DataFrame:
        infected_ids = infected.index
        today = self.dataset.split[curr_date]
        # color is the infector's color, True=purple, False=blue
        contagion_df = pd.concat(
            [
                pd.merge(
                    today, infected["color"], left_on=c, right_index=True, how="inner"
                )
                for c in ("source", "destination")
            ]
        )
        only_newly_infected = contagion_df[
            ~(
                contagion_df["source"].isin(infected_ids)
                & contagion_df["destination"].isin(infected_ids)
            )
        ]
        stacked = only_newly_infected[["source", "destination"]].stack()
        contagion_df = only_newly_infected.join(
            stacked[stacked.isin(infected_ids)]
            .reset_index(drop=True, level=1)
            .rename("infector")
        ).melt(
            id_vars=[
                "datetime",
                self.dataset.infection_param,
                "color",
                "infector",
                "hops",
            ],
            value_name="id",
        )
        contagion_df = contagion_df[~contagion_df["id"].isin(infected_ids)]
        # print(contagion_df)
        if len(contagion_df) == 0:
            return contagion_df
        if self.task["alpha_blue"] < 1:
            contagion_df = self._consider_alpha(contagion_df)
        if self.task["infection_model"] == 1:
            contagion_df = (
                contagion_df.groupby("id")
                .agg({"duration": "sum", "infector": set})
                .pipe(self._cases)
            )
        elif self.task["infection_model"] == 2:
            contagion_df = (
                contagion_df[["id", "duration", "infector", "hops"]]
                .set_index("id")
                .pipe(self._cases)
                .groupby("id")
                .agg({"duration": self._multiply_not_infected_chances, "infector": set})
            )
        contagion_df = contagion_df[self._is_infected(contagion_df["duration"])].drop(
            columns=["duration"]
        )
        return contagion_df


class SQLContagion(Contagion):
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
                set_of_potential_patients, self.task["number_of_patient_zero"]
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
