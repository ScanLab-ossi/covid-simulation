from random import choices, sample
from datetime import datetime, timedelta
from typing import List, Optional
import sys
from pprint import pprint

import numpy as np  # type: ignore
import numpy.ma as ma  # type: ignore
import pandas as pd  # type: ignore
from more_itertools import chunked  # type: ignore

from simulation.helpers import timing
from simulation.state_transition import StateTransition
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.output import Output, Batch
from simulation.constants import *
from simulation.building_blocks import ConnectedBasicBlock, BasicBlock
from simulation.google_cloud import GoogleCloud
from simulation.analysis import Analysis
from simulation.mysql import MySQL


class ContagionRunner(ConnectedBasicBlock):
    """Runs one batch"""

    def run(self, reproducible: bool = False) -> Batch:
        batch = Batch(self.task)
        dt = self.dataset, self.task
        if self.dataset.groups:
            contagion = GroupContagion(*dt, reproducible)
        elif self.dataset.storage == "csv":
            contagion = CSVContagion(*dt, reproducible)
        else:
            contagion = SQLContagion(reproducible=reproducible, gcloud=self.gcloud, *dt)
        # TODO: for-loop that pick the id which start the contagion 
        for i in range(self.task["ITERATIONS"]):
            start = datetime.now()
            output = Output(*dt)
            st = StateTransition(*dt)
            if self.dataset.squeeze > 1:
                period = self.dataset.period // self.dataset.squeeze
                if self.dataset.period % self.dataset.squeeze > 0:
                    period += 1
            elif self.dataset.squeeze < 1:
                period = self.dataset.period * round(self.dataset.squeeze ** -1)
            else:
                period = self.dataset.period + 1
            for day in range(period):
                if day in self.task["patient_zeroes_on_days"]:
                    output.df = output.df.append(
                        contagion.pick_patient_zero(day, output.df.index.tolist())
                    )
                output.df = contagion.contagion(output.df, day).apply(
                    st.move_one, args=(day,), axis=1
                )
                output.value_counts(day)
                if settings["VERBOSE"]:
                    print(f"day {day}")
                    pprint(output.summed[day])
                analysis = Analysis(*dt)
                if settings["INCREMENT"]:
                    if input("continue? y/(n)") != "y":
                        sys.exit()
            batch.append_df(output)
            print(f"iteration {i} took {datetime.now() - start}")
        return batch


class Contagion(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, reproducible: bool = False):
        super().__init__(dataset=dataset, task=task)
        self.reproducible = reproducible
        self.rng = np.random.default_rng(42 if self.reproducible else None)

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
                contagion_df["duration"].values, mask=contagion_df["color"].values,
            )
            * (1 - self.task["alpha_blue"])
        ).data
        new_duration[new_duration > self.task["D_max"]] = self.task["D_max"]
        contagion_df["duration"] = new_duration
        return contagion_df

    def _non_removed(self, df, day):
        return df[
            (df["infection_date"] <= day)
            & df["color"].isin(["green", "purple_red", "purple_pink", "blue"])
        ]

    def _infect(self, contagion_df, day):
        if self.task["alpha_blue"] < 1:
            # FIXME: alpha should be different
            contagion_df = self._consider_alpha(contagion_df)
        if self.task["infection_model"] == 1:
            contagion_df = (
                contagion_df.groupby("susceptible")
                .agg({"duration": "sum", "infector": set})
                .pipe(self._cases)
            )
        elif self.task["infection_model"] == 2:
            contagion_df = (
                contagion_df[
                    ["susceptible", "duration", "infector"]
                    + (["hops"] if self.dataset.hops else [])
                ]
                .set_index("susceptible")
                .pipe(self._cases)
                .groupby("susceptible")
                .agg({"duration": self._multiply_not_infected_chances, "infector": set})
            )
        return (
            contagion_df[self._is_infected(contagion_df["duration"])]
            .drop(columns=["duration"])
            .rename_axis(index=["infected"])
            .assign(**{"infection_date": day, "days_left": 0, "color": "green"})
        )

    @timing
    def _add_age(self, df, random=False):
        if hasattr(self.dataset, "demography"):
            return df.join(self.dataset.demography.set_index("id"), how="left")
        else:
            df["age"] = self.rng.choice(
                list(self.task["age_dist"]),
                len(df),
                p=list(self.task["age_dist"].values()),
            )
        return df


class CSVContagion(Contagion):
    def pick_patient_zero(self, day: int = 0, sick: List[int] = []) -> pd.DataFrame:
        # TODO: pick arbitrary patient
        # today = self.dataset.start_date + timedelta(day) * self.dataset.squeeze
        potential = self.dataset.ids[day]
        if sick:
            potential = list(set(potential) - set(sick))
        n_zero = min(self.task["number_of_patient_zero"], len(potential))
        zeroes = pd.DataFrame(
            [[day, 0, "green"]] * n_zero,
            columns=["infection_date", "days_left", "color"],
            index=self.rng.choice(potential, n_zero, replace=False),
        )  # .pipe(self._add_age)
        return zeroes

    def contagion(self, df: pd.DataFrame, day: int) -> pd.DataFrame:
        infectors = self._non_removed(df, day)
        infector_ids = set(infectors.index)
        today = self.dataset.split[day]
        if self.task.get("max_duration", False):
            today = today[today["duration"] <= self.task["max_duration"]]
        contagion_df = pd.concat(
            [
                pd.merge(
                    today, infectors["color"], left_on=c, right_index=True, how="inner",
                )
                for c in ("source", "destination")
            ]
        )
        infected = contagion_df[
            ~(
                contagion_df["source"].isin(infector_ids)
                & contagion_df["destination"].isin(infector_ids)
            )
        ]
        stacked = infected[["source", "destination"]].stack()
        contagion_df = infected.join(
            stacked[stacked.isin(infector_ids)]
            .reset_index(drop=True, level=1)
            .rename("infector")
        ).melt(
            id_vars=["datetime", "duration", "color", "infector"]
            + (["hops"] if self.dataset.hops else []),
            value_name="susceptible",
        )
        contagion_df = contagion_df[~contagion_df["susceptible"].isin(infector_ids)]
        if len(contagion_df) == 0:
            return df
        df = df.append(contagion_df.pipe(self._infect, day=day).pipe(self._add_age))
        df = df[~df.index.duplicated(keep="first")]
        return df


class GroupContagion(CSVContagion):
    def contagion(self, df: pd.DataFrame, day: int) -> pd.DataFrame:
        infectors = self._non_removed(df, day)
        infector_ids = set(infectors.index)
        today = self.dataset.split[day]
        if self.task.get("max_duration", False):
            today = today[today["duration"] <= self.task["max_duration"]]
        if self.task.get("max_group_size", False):
            today = today[
                today["group"].str.len() <= self.task["max_group_size"]
            ].reset_index(drop=True)
        today["infector"] = today["group"].apply(lambda x: x & infector_ids)
        today = today[today["infector"].str.len() > 0].reset_index()
        today["infector"] = today["infector"].apply(list)
        today["susceptible"] = today["group"].apply(
            lambda x: list(x - set(infector_ids))
        )
        contagion_df = (
            pd.merge(
                today.drop(columns=["susceptible", "group"]).explode("infector"),
                today.drop(columns=["infector", "group"]).explode("susceptible"),
                on=["index", "datetime", "duration"],
                how="outer",
            )
            .dropna(subset=["susceptible"])
            .drop(columns="index")
        )
        if len(contagion_df) == 0:
            return df
        df = df.append(contagion_df.pipe(self._infect, day=day).pipe(self._add_age))
        df = df[~df.index.duplicated(keep="first")]
        return df


class SQLContagion(Contagion):
    def __init__(
        self, dataset: Dataset, task: Task, gcloud: GoogleCloud, reproducible: bool
    ):
        super().__init__(dataset=dataset, task=task, reproducible=reproducible)
        self.mysql = MySQL(gcloud)

    def _squeeze_partitions(self, day: int) -> str:
        r = range(day * self.dataset.squeeze, (day + 1) * self.dataset.squeeze)
        return ", ".join(
            [
                f'{self.dataset.name}_{(self.dataset.start_date + timedelta(d)).strftime("%m%d")}'
                for d in r
                if d < self.dataset.period
            ]
        )

    @timing
    def pick_patient_zero(self, day: int = 0, sick: List[Optional[str]] = []):
        if day == 0 and hasattr(self.dataset, "zeroes") and self.dataset.squeeze == 1:
            potential = self.dataset.zeroes
        else:
            query = f"""SELECT DISTINCT source 
                    FROM datasets.{self.dataset.name} 
                    PARTITION ({self._squeeze_partitions(day)})
                    """
            # if sick:
            #     query += f"WHERE source not in {repr(tuple(sick))}"
            potential = self.mysql.query(query)
            if sick:
                potential = potential[~potential["source"].isin(sick)]
        n_zero = self.task["number_of_patient_zero"]
        zeroes = pd.DataFrame(
            [[day, 0, "green"]] * n_zero,
            columns=["infection_date", "days_left", "color"],
            index=potential["source"].sample(n_zero),
        )  # .pipe(self._add_age)
        return zeroes

    def _make_sql_query(self, infector_ids: List[str], day: int) -> pd.DataFrame:
        extra = ", hops" if self.dataset.hops == True else ""
        query = f"""SELECT source, destination, `datetime`, duration{extra}
                FROM datasets.{self.dataset.name} PARTITION ({self._squeeze_partitions(day)})
                WHERE source in {repr(tuple(infector_ids))}""".replace(
            ",)", ")"
        )
        # AND destination not in {repr(tuple(infector_ids))}"""

        contagion_df = self.mysql.query(query).rename(
            columns={"destination": "susceptible", "source": "infector"}
        )
        contagion_df = contagion_df[
            ~contagion_df["susceptible"].isin(infector_ids)
        ].reset_index(drop=True)
        return contagion_df

    @timing
    def contagion(self, df: pd.DataFrame, day: int) -> pd.DataFrame:
        """
        df : pd.DataFrame
            Columns:
                Name: infection_date, dtype: int64
                Name: days_left, dtype: int64
                Name: color, dtype: object
                Name: age
                Name: infectors?
            Index:
                Name: source, dtype: object
                    ids of infected nodes
        """
        infectors = self._non_removed(df, day)
        infector_ids = list(set(infectors.index))
        if len(infector_ids) == 0:
            return df
        contagion_df = pd.concat(
            [
                self._make_sql_query(list(group), day)
                for group in chunked(infector_ids, 1000)
            ],
        )
        # chunked second param was self.task["number_of_patient_zero"]
        if len(contagion_df) == 0:
            return df
        df = df.append(
            contagion_df.pipe(self._infect, day=day)
        )  # .pipe(self._add_age))
        df = df[~df.index.duplicated(keep="first")]
        return df


"""
today : pd.DataFrame
    Columns:
        Name: datetime, dtype: datetime64[ns]
        Name: duration, dtype: int64
        Name: group, dtype: object
            set of nodes in group meeting
"""
