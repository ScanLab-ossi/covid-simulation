import importlib
from datetime import timedelta
from typing import Dict, List, Union

import pandas as pd
from more_itertools import chunked

from simulation.building_blocks import ConnectedBasicBlock, RandomBasicBlock
from simulation.constants import *
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.helpers import increment, timing
from simulation.mysql import MySQL
from simulation.output import Batch, Output
from simulation.state_transition import StateTransition
from simulation.task import Task


class Contagion(RandomBasicBlock):
    def __init__(self, dataset: Dataset, task: Task, reproducible: bool = False):
        super().__init__(dataset=dataset, task=task, reproducible=reproducible)
        self._get_infection_model()

    def _get_infection_model(self):
        if self.variants.exist:
            self.task["infection_model"] = "VariantInfection"
        elif self.task["infection_model"] == "VariantInfection":
            raise ValueError("Can't use VariantInfection when only one variant exists")
        self.infection_model = getattr(
            importlib.import_module("simulation.infection"),
            self.task["infection_model"],
        )(self.dataset, self.task)

    def _non_removed(self, df: pd.DataFrame) -> set:
        return set(df[df["state"].isin(self.states.infectious_states)].index)

    def _removed(self, df: pd.DataFrame) -> set:
        return set(df[df["state"].isin(self.states.non_infectious_states)].index)

    def _filter_max(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.task.get("max_duration", False):
            df = df[df["duration"] <= self.task["max_duration"]]
        if self.task.get("max_group_size", False):
            df = df[df["group"] <= self.task["max_group_size"]]
        return df

    def soft_filter(self, today: pd.DataFrame) -> pd.DataFrame:
        under_max = self._filter_max(today)
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
        return today


class CSVContagion(Contagion):
    def pick_patient_zero(
        self, variant: str, day: int = 0, sick: List[int] = []
    ) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            index: sick
            columns: infection_date | days_left | state | variant [| age]
        """
        potential = self.dataset.ids[day]
        if sick:
            potential = list(set(potential) - set(sick))
        n_zero = min(
            self.task.get("number_of_patient_zero", variant=variant), len(potential)
        )
        # TODO: days_left should be days in green if Exposed is wanted
        zeroes = pd.DataFrame(
            [[day, 0, "green", variant]] * n_zero,
            columns=["infection_date", "days_left", "state", "variant"],
            index=self.rng.choice(potential, n_zero, replace=False),
        )
        zeroes["variant"] = zeroes["variant"].astype(self.variants.categories)
        return zeroes

    def contagion(
        self,
        infector_df: pd.DataFrame,
        day: int,
        history: Dict[int, str] | None = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        infector_df : pd.DataFrame
            infection_date | days_left | state | variant
        day : int
        """
        today = self.dataset.split[day]
        self._filter_max(today)
        inf, rem = self._non_removed(infector_df), self._removed(infector_df)
        infected = today[
            (~today["destination"].isin(rem) & today["source"].isin(inf))
            ^ (~today["source"].isin(rem) & today["destination"].isin(inf))
        ]
        swapped = infected.rename(
            columns={"source": "destination", "destination": "source"}
        )
        contagion_df = (
            pd.concat([infected, swapped], sort=True, ignore_index=True)
            .rename(columns={"source": "susceptible", "destination": "infector"})
            .join(infector_df[["variant"]], on="infector", how="inner")
            .join(pd.Series(history, name="history", dtype="object"), on="susceptible")
        )
        contagion_df = contagion_df[contagion_df["variant"] != contagion_df["history"]]
        if len(contagion_df) == 0:
            return infector_df
        infector_df = pd.concat(
            [infector_df, self.infection_model._infect(contagion_df, day=day)],
            verify_integrity=True,
        )
        return infector_df


class GroupContagion(CSVContagion):
    # def __init__():
    #     super().__init__(dataset=dataset, task=task)
    #     self.subtract = np.vectorize(lambda x: x - inf - rem)

    # def _intersect(self, x, inf):
    #     return x & inf
    def contagion(self, infector_df: pd.DataFrame, day: int) -> pd.DataFrame:
        """
        Parameters
        ----------
        infector_df : pd.DataFrame
            infection_date | days_left | state | variant
        day : int
        """
        inf, rem = self._non_removed(infector_df), self._removed(infector_df)
        today = self.dataset.split[day]
        today["infector"] = list(map(lambda x: x & inf, today["group"].to_numpy()))
        today["susceptible"] = list(
            map(lambda x: x - inf - rem, today["group"].to_numpy())
        )
        today["group"] = today["group"].str.len()
        today = today[
            (today["infector"].str.len() > 0) & (today["susceptible"].str.len() > 0)
        ].reset_index()
        if len(today) == 0:
            return infector_df
        dupes = (
            today[today["infector"].str.len() > 2]
            .explode("infector")
            .explode("susceptible")
            .join(infector_df[["variant"]], on="infector")
            .groupby(
                ["index", "datetime", "variant", "group", "susceptible"], as_index=False
            )["duration"]
            .sum()
        )
        dupes = dupes[dupes["duration"] > 0]
        today = (
            # today[today["infector"].str.len() == 2]
            today.explode("infector")
            .explode("susceptible")
            .join(infector_df[["variant"]], on="infector")
            .append(dupes)
        )
        if self.task.get("max_duration", False) or self.task.get(
            "max_group_size", False
        ):
            today = self.soft_filter(today)
            if len(today) == 0:
                return infector_df
        infector_df = infector_df.append(
            today.pipe(self.infection_model._infect, day=day),
            verify_integrity=True,
        )
        return infector_df


class SQLContagion(Contagion):
    def __init__(
        self, dataset: Dataset, task: Task, gcloud: GoogleCloud, reproducible: bool
    ):
        super().__init__(dataset=dataset, task=task, reproducible=reproducible)
        self.mysql = MySQL(gcloud)

    def _divide_partitions(self, day: int) -> str:
        r = range(day * self.task["divide"], (day + 1) * self.task["divide"])
        return ", ".join(
            [
                f'{self.dataset.name}_{(self.dataset.start_date + timedelta(d)).strftime("%m%d")}'
                for d in r
                if d < self.dataset.period
            ]
        )

    def pick_patient_zero(self, day: int = 0, sick: List[str] = []):
        # TODO: add variant
        if day == 0 and hasattr(self.dataset, "zeroes") and self.task["divide"] == 1:
            potential = self.dataset.zeroes
        else:
            query = f"""SELECT DISTINCT source 
                    FROM datasets.{self.dataset.name} 
                    PARTITION ({self._divide_partitions(day)})
                    """
            # if sick:
            #     query += f"WHERE source not in {repr(tuple(sick))}"
            potential = self.mysql.query(query)
            if sick:
                potential = potential[~potential["source"].isin(sick)]
        n_zero = self.task["number_of_patient_zero"]
        zeroes = pd.DataFrame(
            [[day, 0, "green"]] * n_zero,
            columns=["infection_date", "days_left", "state"],
            index=potential["source"].sample(n_zero),
        )
        return zeroes

    def _make_sql_query(self, infector_ids: List[str], day: int) -> pd.DataFrame:
        extra = ", hops" if self.dataset.hops == True else ""
        query = f"""SELECT source, destination, `datetime`, duration{extra}
                FROM datasets.{self.dataset.name} PARTITION ({self._divide_partitions(day)})
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
                Name: state, dtype: object
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
        df = df.append(contagion_df.pipe(self.infection_model._infect, day=day))
        df = df[~df.index.duplicated(keep="first")]
        return df


class ContagionRunner(ConnectedBasicBlock):
    """Runs one batch"""

    def _get_patient_zero(
        self,
        day: int,
        output: Output,
        contagion: Union[CSVContagion, GroupContagion, SQLContagion],
    ) -> Output:
        for variant in self.task["variants"]:
            if day in self.task.get("patient_zeroes_on_days", variant=variant):
                # TODO: do you need to explicitly loop for each variant? or can it all be done at once?
                output.df = pd.concat(
                    [
                        output.df,
                        contagion.pick_patient_zero(
                            variant=variant, day=day, sick=output.df.index.tolist()
                        ),
                    ]
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

    def _reinfect(self, output: Output) -> Output:
        indexer = (output.df["state"] == "white") & (
            output.df["days_left"] == 0
        ).sort_index()
        if self.task["reinfect"] > 1:
            removed = output.history.keys() & set(indexer.index)
            output.df = output.df.drop(removed)
            indexer = indexer.drop(removed)
        output.history |= output.df[indexer]["variant"].to_dict()
        output.df = output.df[~indexer]
        return output

    @timing
    def run(self) -> Batch:
        self.dataset.load_dataset()
        dt = self.dataset, self.task
        batch = Batch(*dt)
        contagion = self._pick_contagion(reproducible=False)
        for _ in range(self.task["ITERATIONS"]):
            output, st = Output(*dt), StateTransition(*dt)
            for day in range(max(self.dataset.split) + 1):
                output = self._get_patient_zero(day, output, contagion)
                output.df = (
                    contagion.contagion(output.df, day, output.history)
                    .apply(st.move_one, axis=1)
                    .pipe(self.variants.categorify)
                    .sort_index()
                )
                if self.task["reinfect"] > 0:
                    output = self._reinfect(output)
                output.value_counts(day)
                increment()
                if settings["VERBOSE"]:
                    print(f"did day {day}")
            batch.append_output(output)
        return batch
