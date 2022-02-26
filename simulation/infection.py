from doctest import DocFileTest
from itertools import groupby
from math import exp, log
from typing import List

import numpy as np
import pandas as pd
from more_itertools import powerset
from pandas.core.frame import DataFrame

from simulation.building_blocks import RandomBasicBlock
from simulation.constants import *
from simulation.helpers import timing


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
            datetime | infector | duration | susceptible | variant
        day : int

        Returns
        -------
        pd.DataFrame
            index : infected
            columns : infection_date | days_left | state | variant

    """

    def _organize(self, contagion_df: pd.DataFrame, day: int):
        contagion_df = contagion_df[self.variants.column]
        infected = (
            contagion_df.rename_axis(index=["infected"])
            .assign(**{"infection_date": day, "days_left": 0, "state": "green"})
            .pipe(self.states.categorify)
        )
        return infected

    def _is_infected(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df
        f = lambda x: self.rng.choice([True, False], 1, p=[x, 1 - x])
        return df[np.array(list(map(f, df["duration"].to_numpy())))].copy()


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
    # 2 and 4
    def _mult(self, s: pd.Series) -> float:
        res = 1 - np.prod(
            1 - np.minimum(s.to_numpy() / self.task["D_max"], 1) * self.task["P_max"]
        )
        return res

    def _cases(self, df: pd.DataFrame) -> pd.DataFrame:
        df["duration"] = np.where(
            df["duration"].to_numpy() >= self.task.get("D_min"),
            df["duration"].to_numpy(),
            0.00001,
        )
        return df

    def _infect(self, contagion_df: pd.DataFrame, day: int) -> pd.DataFrame:
        contagion_df = self._cases(contagion_df)
        contagion_df = contagion_df.groupby(["susceptible"] + self.variants.column).agg(
            {"duration": self._mult}
        )
        if self.variants:
            contagion_df = contagion_df.reset_index(level="variant").dropna()
        contagion_df = contagion_df.pipe(self._is_infected)
        contagion_df = contagion_df.pipe(self._organize, day)
        return contagion_df


class SigmoidInfection(GroupInfection):
    # 3
    def _mult(self, s: pd.Series) -> float:
        np.seterr(over="raise", divide="raise")
        mu = (self.task["D_max"] + self.task["D_min"]) / 2
        P_max = self.task["P_max"]
        if self.task["skew"] == 0:
            self.task["skew"] = 0.1
        for _ in range(10):
            skew = self.task["skew"]
            try:
                res = 1 - np.prod(1 - P_max / (1 + np.exp((mu - s.to_numpy()) / skew)))
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
    @timing
    def _mult(self, df: pd.DataFrame) -> pd.DataFrame:
        for p in ["P_max", "D_max"]:
            df[p] = df["variant"].astype(str).replace(self.variants.variant_to_param(p))
        if self.task.get("reinfect") > 0 and len(self.variants) == 2:
            immunity = {
                k: round(1 - v, 4)
                for k, v in self.variants.variant_to_param("immunity").items()
            }
            history_col = (
                df[["susceptible", "history"]]
                .drop_duplicates("susceptible")
                .set_index("susceptible")
                .replace(immunity)
                .fillna(1)
                .sort_index()
                .to_numpy()
            )
        df = (
            df.groupby(["susceptible", "variant"])
            .apply(
                lambda g: 1
                - np.prod(1 - np.minimum(g["duration"] / g["D_max"], 1) * g["P_max"])
            )
            .unstack(level=1, fill_value=0)
        )
        # not a great solution
        if self.task.get("reinfect") > 0 and len(self.variants) == 2:
            df = df.mul(history_col, axis="rows")
        for v in set(self.variants) - set(df.columns):
            df[v] = 0
        return df

    def _cases(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, "duration"] = df["duration"] * df["variant"].astype(str).replace(
            {k: v * 0.1 + 1 for k, v in self.variants.variant_to_param("j").items()}
        )
        diff_d_min = (
            df["variant"]
            .apply(lambda x: self.task.get("D_min", variant=x))
            .astype("int")
        )
        df["duration"] = np.where(
            df["duration"].to_numpy() >= diff_d_min,
            df["duration"].to_numpy(),
            0.00001,
        )
        return df

    # def _zip(self, df: pd.DataFrame) -> pd.DataFrame:
    #     df["duration"] = list(zip(df["duration"], df["variant"]))
    #     return df

    @timing
    def _infect(self, contagion_df: pd.DataFrame, day: int) -> pd.DataFrame:
        contagion_df = contagion_df.pipe(self._cases)
        contagion_df = contagion_df.pipe(self._mult)
        contagion_df["duration"] = contagion_df.apply(self._inclusion_exclusion, axis=1)
        contagion_df = contagion_df.pipe(self._is_infected)[self.variants]
        # if len(contagion_df) == 0:
        # return contagion_df.pipe(self._organize, day=day)
        contagion_df = contagion_df.pipe(self._normalize_variants)
        contagion_df = contagion_df.pipe(self._choose_variant)
        contagion_df = contagion_df.pipe(self._organize, day=day)
        return contagion_df

    def _inclusion_exclusion(self, p: List[float]) -> float:
        # P(not-infected)
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
        df = (df.T / df.sum(axis="columns")).T
        return df

    def _choose_variant(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame([], columns=["variant"])
        df["variant"] = df.apply(lambda x: self.rng.choice(self.variants, p=x), axis=1)
        return self.variants.categorify(df)
