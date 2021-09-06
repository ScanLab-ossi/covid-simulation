from itertools import groupby
from math import exp, log
from typing import List

import numpy as np
import pandas as pd
from more_itertools import powerset

from building_blocks import RandomBasicBlock
from constants import *
from helpers import timing


class Infection(RandomBasicBlock):
    """
    _cases():

        Parameters
        ----------
        df : pd.DataFrame
            datetime | duration | susceptible | variant [| hops]

        Returns
        -------
        pd.DataFrame
            datetime | duration | susceptible | variant [| hops]

    _infect():

        Parameters
        ----------
        contagion_df : pd.DataFame
            datetime | duration | susceptible | variant
        day : int

        Returns
        -------
        pd.DataFrame
            index : infected
            columns : infection_date | days_left | color | variant

    """

    def _organize(self, contagion_df: pd.DataFrame, day: int):
        infected = (
            contagion_df.drop(columns=["duration"])
            .rename_axis(index=["infected"])
            .assign(**{"infection_date": day, "days_left": 0, "color": "green"})
        )
        infected["color"] = infected["color"].astype(self.states.categories("states"))
        return infected

    def _is_infected(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df
        vec_choice = np.vectorize(
            lambda x: self.rng.choice([True, False], 1, p=[x, 1 - x])
        )
        return df[vec_choice(df["duration"].to_numpy())]


class AdditiveInfection(Infection):
    def _cases(self, df: pd.DataFrame) -> pd.DataFrame:
        df["duration"] = np.where(
            df["duration"].to_numpy() >= self.task["D_min"],
            np.minimum(
                df["duration"].to_numpy() / self.task["D_max"] * self.task["P_max"],
                1.0,
            ),
            0,
        )

    def _infect(self, contagion_df: pd.DataFrame, day: int) -> pd.DataFrame:
        contagion_df = (
            contagion_df.groupby("susceptible")
            .agg({"duration": "sum", "infector": set})
            .pipe(self._cases)
            .pipe(self._is_infected)
        )
        return self._organize(contagion_df, day)


class GroupInfection(Infection):
    # 2 and 3
    def _multiply_not_infected_chances(self, d_i_k: pd.Series) -> float:
        res = 1 - np.prod(
            1
            - np.minimum(d_i_k.to_numpy() / self.task["D_max"], 1) * self.task["P_max"]
        )
        return res

    def _cases(self, df: pd.DataFrame) -> pd.DataFrame:
        hops = df["hops"].to_numpy() if self.dataset.hops else 1
        df["duration"] = np.where(
            df["duration"].to_numpy() >= self.task["D_min"],
            df["duration"].to_numpy() / hops,
            0.00001,
        )
        return df

    def _infect(self, contagion_df: pd.DataFrame, day: int) -> pd.DataFrame:
        contagion_df = (
            contagion_df.pipe(self._cases)
            .groupby(["susceptible", "variant"])
            .agg({"duration": self._multiply_not_infected_chances})
            .reset_index(level="variant")
            .dropna()
            .pipe(self._is_infected)
            .pipe(self._organize, day)
        )
        return contagion_df


class SigmoidInfection(GroupInfection):
    # 3
    def _multiply_not_infected_chances(self, d_i_k: pd.Series) -> float:
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


class ConstantRateInfection(Infection):
    # 5
    def _infect(self, contagion_df: pd.DataFrame, day: int) -> pd.DataFrame:
        contagion_df["duration"] = self.task["beta"]
        contagion_df = (
            contagion_df.pipe(self._is_infected)
            .drop(columns=["infector", "datetime"])
            .set_index("susceptible")
        )
        return self._organize(contagion_df, day)


class VariantInfection(GroupInfection):
    # 4
    def _cases(self, df: pd.DataFrame) -> pd.DataFrame:
        df["duration"] = df["duration"] * df["variant"].astype(str).apply(
            lambda v: self.task[v]["j"] * 0.1 + 1
        )
        df["duration"] = np.where(
            df["duration"].to_numpy() >= self.task["D_min"],
            df["duration"].to_numpy(),
            0.00001,
        )
        return df

    def _infect(self, contagion_df: pd.DataFrame, day: int) -> pd.DataFrame:
        contagion_df = (
            contagion_df.pipe(self._cases)
            .groupby(["susceptible", "variant"])
            .agg({"duration": self._multiply_not_infected_chances})
            .droplevel("variant")
            .fillna(0)
            .groupby(["susceptible"])
            .agg(
                variant=("duration", list),
                duration=("duration", self._inclusion_exclusion),
            )
            .pipe(self._is_infected)
            .pipe(self._normalize_variants)
            .pipe(self._choose_variant)
            .pipe(self._organize, day=day)
        )
        return contagion_df

    def _inclusion_exclusion(self, p: List[float]) -> float:
        powerset_ = [list(x) + [np.prod(x)] for x in list(powerset(p))[1:]]
        return min(
            sum(
                [
                    np.sum((-1) ** k_plus_1 * np.array([x[-1] for x in group]))
                    for k_plus_1, group in groupby(powerset_, len)
                ]
            ),
            1,
        )

    def _normalize_variants(self, df: pd.DataFrame) -> pd.DataFrame:
        df["variant"] = df["variant"].apply(np.array) / df["variant"].apply(sum)
        return df

    def _choose_variant(self, df: pd.DataFrame) -> pd.DataFrame:
        df["variant"] = (
            df["variant"]
            .apply(lambda x: self.rng.choice(self.variants.list, 1, p=x)[0])
            .astype(self.variants.categories)
        )
        return df
