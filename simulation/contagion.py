import sys
from datetime import datetime, timedelta
from math import prod
from typing import List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from more_itertools import chunked, powerset  # type: ignore

from simulation.analysis import Analysis
from simulation.building_blocks import BasicBlock, ConnectedBasicBlock
from simulation.constants import *
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.helpers import timing
from simulation.mysql import MySQL
from simulation.output import Batch, Output
from simulation.state_transition import StateTransition
from simulation.task import Task


class ContagionRunner(ConnectedBasicBlock):
    """Runs one batch"""

    def _squeeze(self) -> int:
        if self.dataset.squeeze > 1:
            period = self.dataset.period // self.dataset.squeeze
            if self.dataset.period % self.dataset.squeeze > 0:
                period += 1
        elif self.dataset.squeeze < 1:
            period = self.dataset.period * round(self.dataset.squeeze ** -1)
        else:
            period = self.dataset.period + 1
        return period

    def run(self, reproducible: bool = False) -> Batch:
        dt = self.dataset, self.task
        batch = Batch(*dt)
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
            for day in range(self._squeeze()):
                for variant in self.task.variants():
                    if day in self.task[variant]["patient_zeroes_on_days"]:
                        # TODO: do you need to explicitly loop for each variant? or can it all be done at once?
                        output.df = output.df.append(
                            contagion.pick_patient_zero(
                                variant=variant, day=day, sick=output.df.index.tolist()
                            )
                        )
                output.df = contagion.contagion(output.df, day).apply(
                    st.move_one, args=(day,), axis=1
                )
                output.value_counts(day)
                if settings["VERBOSE"]:
                    pass
                    # print(f"day {day}")
                    # pprint(output.summed[day])
                # analysis = Analysis(*dt)
                if settings["INCREMENT"]:
                    if input("continue? y/(n)") != "y":
                        sys.exit()
            batch.append_output(output)
            print(f"iteration {i} took {datetime.now() - start}")
        return batch


class Contagion(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, reproducible: bool = False):
        super().__init__(dataset=dataset, task=task)
        self.reproducible = reproducible
        self.rng = np.random.default_rng(42 if self.reproducible else None)

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
                df["duration"].values >= self.task["D_min"],
                np.minimum(
                    df["duration"].values / self.task["D_max"] * self.task["P_max"], 1.0
                ),
                0,
            )
        elif self.task["infection_model"] in (2, 3):
            hops = df["hops"].values if self.dataset.hops else 1
            df["duration"] = np.where(
                df["duration"].values >= self.task["D_min"],
                df["duration"].values / hops,
                0.00001,
            )
        elif self.task["infection_model"] == 4:
            df["duration"] = df["duration"] * df["variant"].apply(
                lambda v: self.task[v]["j"] * 0.1 + 1
            )
            df["duration"] = np.where(
                df["duration"].values >= self.task["D_min"],
                df["duration"].values,
                0.00001,
            )
        return df

    def _is_infected(self, df: pd.DataFrame) -> pd.DataFrame:
        vec_choice = np.vectorize(
            lambda x: self.rng.choice([True, False], 1, p=[x, 1 - x])
        )
        return df[vec_choice(df["duration"].values)]

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
                        1 - P_max / (1 + np.exp((mu - d_i_k.values) / skew))
                    )
                    break
                except FloatingPointError:
                    self.task["skew"] *= 10
            return res
        elif self.task["infection_model"] in (2, 4):
            res = 1 - np.prod(
                1
                - np.minimum(d_i_k.values / self.task["D_max"], 1) * self.task["P_max"]
            )
            return res

    def _non_removed(self, df, day):
        return df[
            (df["infection_date"] <= day)
            & df["color"].isin(["green", "purple_red", "purple_pink", "blue"])
        ]

    def _normalize_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("duration")
        for i in range(len(df["duration"]) - 1):
            curr_dist = df["duration"].iloc[i:].tolist()
            this = df["duration"].iloc[i]
            df.iloc[i, df.columns.get_loc("duration")] = this - sum(
                [prod(x) for x in powerset(curr_dist) if len(x) > 1 and this in x]
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
                .agg(
                    {"duration": self._multiply_not_infected_chances}
                )  # drops infector
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
        infected = (
            contagion_df.drop(columns=["duration"])
            .rename_axis(index=["infected"])
            .assign(**{"infection_date": day, "days_left": 0, "color": "green"})
        )
        return infected

    @timing
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
    @timing
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
        # TODO:
        # today = self.dataset.start_date + timedelta(day) * self.dataset.squeeze
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
        infectors = self._non_removed(df, day)
        infector_ids = set(infectors.index)
        today = self.dataset.split[day]
        if self.task.get("max_duration", False):
            today = today[today["duration"] <= self.task["max_duration"]]
        contagion_df = pd.concat(
            [
                pd.merge(
                    today,
                    infectors["color"],
                    left_on=c,
                    right_index=True,
                    how="inner",
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
    @timing
    def contagion(self, df: pd.DataFrame, day: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            infection_date | days_left | color | variant
        day : int
        """
        infector_ids = set(self._non_removed(df, day).index)
        today = self.dataset.split[day]
        today["infector"] = today["group"].apply(lambda x: list(x & infector_ids))
        today = today[today["infector"].str.len() > 0].reset_index()
        if len(today) == 0:
            return df
        today["susceptible"] = today["group"].apply(
            lambda x: list(x - set(infector_ids))
        )
        today = today[today["susceptible"].str.len() > 0]
        if len(today) == 0:
            return df
        today = today.explode("infector").join(df[["variant"]], on="infector")
        filtered = today.pipe(self._filter_max)
        # TODO: add option for strict filtering (remove), in addition to current light filtering (break up)
        non_filtered = today[~today.index.isin(filtered)].drop(
            columns=["group", "index"]
        )
        non_filtered["infector"] = non_filtered["infector"].apply(lambda x: (x,))
        filtered = (
            filtered.groupby(["index", "variant"])
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
        today = (
            filtered.append(non_filtered).explode("susceptible").reset_index(drop=True)
        )
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
        # TODO: add variant
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
"""
    @timing
    def contagion(self, df: pd.DataFrame, day: int) -> pd.DataFrame:
        ""
        Parameters
        ----------
        df : pd.DataFrame
            infection_date | days_left | color | variant
        day : int
        ""
        infectors = self._non_removed(df, day)
        infector_ids = set(infectors.index)
        today = self.dataset.split[day]
        # .pipe(self._filter_max)
        today["infector"] = today["group"].apply(lambda x: list(x & infector_ids))
        today = today[today["infector"].str.len() > 0].reset_index()
        if len(today) == 0:
            return df
        today["susceptible"] = today["group"].apply(
            lambda x: list(x - set(infector_ids))
        )
        today = pd.concat(today[today["susceptible"].str.len() > 0].apply(self.sum_groups, axis=1).tolist())
        if len(today) == 0:
            return df
        contagion_df = (
            .groupby(["index", "variant"])
            .agg(
                {
                    "duration": "sum",
                    "infector": tuple,
                    "susceptible": "first",
                    "datetime": "first",
                }
            )
            .reset_index("variant")
            .explode("susceptible")
            .reset_index(drop=True)
        )
        if len(contagion_df) == 0:
            return df
        df = df.append(contagion_df.pipe(self._infect, day=day))
        # .pipe(self._add_age))
        df = df[~df.index.duplicated(keep="first")]
        return df
    def sum_groups(self, s: pd.Series, infectors: pd.DataFrame) -> pd.DataFrame:
        res = (
            infectors[infectors["index"].isin(s["infector"])]
            .groupby("variant")
            .agg(
                **{
                    "duration": pd.NamedAgg(column="variant", aggfunc="count"),
                    "infector": pd.NamedAgg(column="index", aggfunc="first"),
                }
            )
        ).reset_index()
        res["susceptible"] = [s["susceptible"]] * len(res)
        res["duration"] *= s["duration"]
        res["index"] = s["index"]
        res["infector"] = res["infector"].str[0]
        return res
"""