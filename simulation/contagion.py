import sys
from datetime import datetime, timedelta
from typing import List, Optional, Union
from math import exp, log

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from more_itertools import chunked, powerset  # type: ignore

from simulation.building_blocks import RandomBasicBlock, ConnectedBasicBlock
from simulation.constants import *
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.helpers import timing
from simulation.mysql import MySQL
from simulation.output import Batch, Output
from simulation.state_transition import StateTransition
from simulation.task import Task


class Contagion(RandomBasicBlock):
    def _cases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            datetime | duration | infector | susceptible | variant [| hops]

        Returns
        -------
        pd.DataFrame
            datetime | duration | infector | susceptible | variant [| hops]
        """

        if self.task["infection_model"] == 1:
            df["duration"] = np.where(
                df["duration"].to_numpy() >= self.task["D_min"],
                np.minimum(
                    df["duration"].to_numpy() / self.task["D_max"] * self.task["P_max"],
                    1.0,
                ),
                0,
            )
        elif self.task["infection_model"] in (2, 3):
            hops = df["hops"].to_numpy() if self.dataset.hops else 1
            df["duration"] = np.where(
                df["duration"].to_numpy() >= self.task["D_min"],
                df["duration"].to_numpy() / hops,
                0.00001,
            )
        elif self.task["infection_model"] == 4:
            df["duration"] = df["duration"] * df["variant"].apply(
                lambda v: self.task[v]["j"] * 0.1 + 1
            )
            df["duration"] = np.where(
                df["duration"].to_numpy() >= self.task["D_min"],
                df["duration"].to_numpy(),
                0.00001,
            )
        return df

    def _is_infected(self, df: pd.DataFrame) -> pd.DataFrame:
        vec_choice = np.vectorize(
            lambda x: self.rng.choice([True, False], 1, p=[x, 1 - x])
        )
        return df[vec_choice(df["duration"].to_numpy())]

    def _multiply_not_infected_chances(self, d_i_k: pd.Series) -> float:
        if self.task["infection_model"] == 3:
            np.seterr(over="raise", divide="raise")
            mu = (self.task["D_max"] + self.task["D_min"]) / 2
            P_max = self.task["P_max"]
            if self.task["skew"] == 0:
                self.task["skew"] = 0.1
            for _ in range(10):
                skew = self.task["skew"]
                try:
                    res = 1 - np.prod(
                        1 - P_max / (1 + np.exp((mu - d_i_k.to_numpy()) / skew))
                    )
                    break
                except FloatingPointError:
                    self.task["skew"] *= 10
            return res
        elif self.task["infection_model"] in (2, 4):
            res = 1 - np.prod(
                1
                - np.minimum(d_i_k.to_numpy() / self.task["D_max"], 1)
                * self.task["P_max"]
            )
            return res

    def _non_removed(self, df: pd.DataFrame) -> set:
        return set(
            df[df["color"].isin(["green", "purple_red", "purple_pink", "blue"])].index
        )

    def _normalize_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("duration")
        for i in range(len(df["duration"]) - 1):
            curr_dist = df["duration"].iloc[i:].tolist()
            this = df["duration"].iloc[i]
            df.iloc[i, df.columns.get_loc("duration")] = this - sum(
                [
                    exp(sum(map(log, x)))
                    for x in powerset(curr_dist)
                    if len(x) > 1 and this in x
                ]
            )
        df["duration"] = df["duration"] * 1 / df["duration"].sum()
        return df

    def _which(self, df: pd.DataFrame) -> pd.Series:
        return self.rng.choice(df, p=df["duration"].tolist())

    def _which_variant(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(
            [
                self._which(self._normalize_variants(dd))
                for _, dd in df.groupby("susceptible")
            ],
            columns=df.columns,
        )
        return df

    def _is_infected_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        # FIXME: not sum, union!
        durs = df.groupby("susceptible").sum() - df.groupby("susceptible").mult()
        durs2 = durs.sum("duration")
        df = df[
            df["susceptible"].isin(
                durs2[durs2["duration"] > self.rng.random(len(durs2))].index
            )
        ]
        return df

    def _infect(self, contagion_df: pd.DataFrame, day: int) -> pd.DataFrame:
        """
        infect

        Parameters
        ----------
        contagion_df : pd.DataFame
            datetime | duration | infector | susceptible | variant
        day : int

        Returns
        -------
        pd.DataFrame
            index : infected
            columns : infector | infection_date | days_left | color | variant

        """
        if self.task["infection_model"] == 1:
            contagion_df = (
                contagion_df.groupby("susceptible")
                .agg({"duration": "sum", "infector": set})
                .pipe(self._cases)
                .pipe(self._is_infected)
            )
        elif self.task["infection_model"] in (2, 3):
            contagion_df = (
                contagion_df.set_index("susceptible")
                .pipe(self._cases)
                .groupby(["susceptible", "variant"])
                .agg({"duration": self._multiply_not_infected_chances})
                .reset_index(level="variant")
                .pipe(self._is_infected)
            )
        elif self.task["infection_model"] == 4:
            contagion_df = (
                contagion_df.pipe(self._cases)
                .groupby(["susceptible", "variant"])
                .agg(
                    {"duration": self._multiply_not_infected_chances}
                )  # drops infector
                .reset_index(level="variant")
                .pipe(self._is_infected_variants)
                .pipe(self._which_variant)
            )
        elif self.task["infection_model"] == 5:
            contagion_df["duration"] = self.task["beta"]
            contagion_df = (
                contagion_df.pipe(self._is_infected)
                .drop(columns=["infector", "datetime"])
                .set_index("susceptible")
            )
        infected = (
            contagion_df.drop(columns=["duration"])
            .rename_axis(index=["infected"])
            .assign(**{"infection_date": day, "days_left": 0, "color": "green"})
        )
        return infected

    def _add_age(self, df: pd.DataFrame, random: bool = False) -> pd.DataFrame:
        if hasattr(self.dataset, "demography"):
            return df.join(self.dataset.demography.set_index("id"), how="left")
        else:
            df["age"] = self.rng.choice(
                list(self.task["age_dist"]),
                len(df),
                p=list(self.task["age_dist"].values()),
            )
        return df

    def _filter_max(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.task.get("max_duration", False):
            df = df[df["duration"] <= self.task["max_duration"]]
        if self.task.get("max_group_size", False):
            df = df[df["group"].str.len() <= self.task["max_group_size"]]
        return df


class CSVContagion(Contagion):
    def pick_patient_zero(
        self, variant: str, day: int = 0, sick: List[int] = []
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            index: sick
            columns: infection_date | days_left | color | variant [| age]
        """
        # TODO: pick arbitrary patient
        # today = self.dataset.start_date + timedelta(day) * self.task['squeeze']
        potential = self.dataset.ids[day]
        if sick:
            potential = list(set(potential) - set(sick))
        n_zero = min(self.task[variant]["number_of_patient_zero"], len(potential))
        zeroes = pd.DataFrame(
            [[day, 0, "green", variant]] * n_zero,
            columns=["infection_date", "days_left", "color", "variant"],
            index=self.rng.choice(potential, n_zero, replace=False),
        )  # .pipe(self._add_age)
        return zeroes

    def contagion(self, df: pd.DataFrame, day: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            infection_date | days_left | color | variant
        day : int
        """
        infector_ids = self._non_removed(df)
        # FIXME: broken cuz _non_removed returns a set and not a df
        today = self.dataset.split[day]
        has_duration = "duration" in today.columns
        if self.task.get("max_duration", False):
            today = today[today["duration"] <= self.task["max_duration"]]
        contagion_df = today[
            (
                today["source"].isin(infector_ids)
                | today["destination"].isin(infector_ids)
            )
        ]
        infected = contagion_df[
            ~(
                contagion_df["source"].isin(infector_ids)
                & contagion_df["destination"].isin(infector_ids)
            )
        ]
        stacked = infected[["source", "destination"]].stack()
        id_vars = list(set(infected.columns) & {"color", "duration", "hops"})
        contagion_df = (
            infected.join(
                stacked[stacked.isin(infector_ids)]
                .reset_index(drop=True, level=1)
                .rename("infector")
            )
            .melt(
                id_vars=["datetime", "infector"] + id_vars,
                value_name="susceptible",
            )
            .drop(columns=["variable"])
        )
        contagion_df["variant"] = "variant_a"  # FIXME!!!!
        contagion_df = contagion_df[~contagion_df["susceptible"].isin(infector_ids)]
        if len(contagion_df) == 0:
            return df
        df = df.append(contagion_df.pipe(self._infect, day=day))
        df = df[~df.index.duplicated(keep="first")]
        return df


class GroupContagion(CSVContagion):
    def contagion(self, df: pd.DataFrame, day: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            infection_date | days_left | color | variant
        day : int
        """
        infector_ids = self._non_removed(df)
        today = self.dataset.split[day]
        intersect = np.vectorize(lambda x: x & infector_ids)
        subtract = np.vectorize(lambda x: x - infector_ids)
        today["infector"] = intersect(today["group"].to_numpy())
        today["susceptible"] = subtract(today["group"].to_numpy())
        today = today[
            (today["infector"].str.len() > 0) & (today["susceptible"].str.len() > 0)
        ].reset_index()
        if len(today) == 0:
            return df
        today = today.explode("infector").join(df[["variant"]], on="infector")
        under_max = today.pipe(self._filter_max)
        # TODO: add option for strict filtering (remove), in addition to current light filtering (break up)
        over_max = today[~today["index"].isin(under_max["index"])].drop(
            columns=["group", "index"]
        )
        over_max["infector"] = over_max["infector"].apply(lambda x: (x,))
        under_max = (
            under_max.groupby(["index", "variant"])
            .agg(
                {
                    "duration": "sum",
                    "infector": tuple,
                    "susceptible": "first",
                    "datetime": "first",
                }
            )
            .reset_index("variant")
        )
        today = under_max.append(over_max).explode("susceptible").reset_index(drop=True)
        if len(today) == 0:
            return df
        df = df.append(today.pipe(self._infect, day=day))
        # .pipe(self._add_age))
        df = df[~df.index.duplicated(keep="first")]
        return df


class SQLContagion(Contagion):
    def __init__(
        self, dataset: Dataset, task: Task, gcloud: GoogleCloud, reproducible: bool
    ):
        super().__init__(dataset=dataset, task=task, reproducible=reproducible)
        self.mysql = MySQL(gcloud)

    def _squeeze_partitions(self, day: int) -> str:
        r = range(day * self.task["squeeze"], (day + 1) * self.task["squeeze"])
        return ", ".join(
            [
                f'{self.dataset.name}_{(self.dataset.start_date + timedelta(d)).strftime("%m%d")}'
                for d in r
                if d < self.dataset.period
            ]
        )

    def pick_patient_zero(self, day: int = 0, sick: List[Optional[str]] = []):
        # TODO: add variant
        if day == 0 and hasattr(self.dataset, "zeroes") and self.task["squeeze"] == 1:
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
        infector_ids = list(self._non_removed(df))
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


class ContagionRunner(ConnectedBasicBlock):
    """Runs one batch"""

    # def _squeeze(self) -> int:
    #     if self.task["squeeze"] > 1:
    #         period = max(self.dataset.split)
    #         if self.dataset.period % self.task["squeeze"] > 0:
    #             period += 1
    #     elif self.task["squeeze"] < 1:
    #         period = self.dataset.period * round(self.task["squeeze"] ** -1)
    #     else:
    #         period = self.dataset.period + 1
    #     return period

    def _get_patient_zero(
        self,
        day: int,
        output: Output,
        contagion: Union[CSVContagion, GroupContagion, SQLContagion],
    ) -> Output:
        for variant in self.task.variants():
            if day in self.task[variant]["patient_zeroes_on_days"]:
                # TODO: do you need to explicitly loop for each variant? or can it all be done at once?
                output.df = output.df.append(
                    contagion.pick_patient_zero(
                        variant=variant, day=day, sick=output.df.index.tolist()
                    )
                )
        return output

    def _pick_contagion(
        self, reproducible: bool
    ) -> Union[CSVContagion, GroupContagion, SQLContagion]:
        if self.dataset.groups:
            return GroupContagion(self.dataset, self.task, reproducible)
        elif self.dataset.storage == "csv":
            return CSVContagion(self.dataset, self.task, reproducible)
        else:
            return SQLContagion(
                self.dataset, self.task, reproducible=reproducible, gcloud=self.gcloud
            )

    def run(self) -> Batch:
        dt = self.dataset, self.task
        batch = Batch(*dt)
        contagion = self._pick_contagion(reproducible=False)
        for i in range(self.task["ITERATIONS"]):
            start, output, st = datetime.now(), Output(*dt), StateTransition(*dt)
            for day in range(max(self.dataset.split) + 1):
                output = self._get_patient_zero(day, output, contagion)
                if day == 0:
                    output.set_first()
                output.df = contagion.contagion(output.df, day).apply(
                    st.move_one, axis=1
                )
                output.value_counts(day)
                if settings["INCREMENT"]:
                    if input("continue? y/(n)") != "y":
                        sys.exit()
            output.set_damage_assessment()
            batch.append_output(output)
            print(f"iteration {i} took {datetime.now() - start}")
        return batch


"""
today : pd.DataFrame
    Columns:
        Name: datetime, dtype: datetime64[ns]
        Name: duration, dtype: int64
        Name: group, dtype: object
            set of nodes in group meeting
"""