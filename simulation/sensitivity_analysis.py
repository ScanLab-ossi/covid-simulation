from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal
from math import comb
from itertools import product

import numpy as np
import pandas as pd

from simulation.building_blocks import BasicBlock, ConnectedBasicBlock
from simulation.constants import *
from simulation.contagion import ContagionRunner
from simulation.helpers import timing
from simulation.output import MultiBatch

if TYPE_CHECKING:
    from output import Batch


class Analysis(BasicBlock):
    """
    Sums a Batch of results for sensitivity analysis.

    Methods
    -------
    count()


    Inherited
    ---------
    Attributes: task, dataset, states
    """

    def count(
        self,
        df: pd.DataFrame,
        grouping: str,
        threshold: int | float | None = None,
        specific_day: int | None = None,
        percent: bool = True,
        how: Literal["amount", "day"] = "amount",
        cumsum: bool = False,
    ) -> List[str]:
        """Sums average results of batch of iterations, according to parameters given.

        Parameters
        ----------
        df : pd.DataFrame
        grouping : str
            any state in States, not_{state}, daily_{state}
        threshold : Union[int, float]
            amount or percent over which to return day at which arrived at threshold
        specific_day: int | None
            day on which to check metric
        percent : bool
            return res in percent if True, regular numerical amount if False
        how : str [amount, day]
            *amount* or *day* that the conditions are filled

        Raises
        ------
        NotImplementedError
            If threshold is set but how is specified.
        """
        # amount / day @ max percent / amount of color / sick
        # day @ specific percent / amount of color / sick
        metric_name = f"{'max' if not threshold else 'day_of_specific'}_{'percent'if percent else how}_{grouping}"
        if "variant" in df.columns:
            df = df.drop(columns=["variant"])
        if grouping not in df.columns:
            df = df.pipe(self.sum_groupings, how=grouping)
        if percent:
            df = df / self.dataset.nodes
        if cumsum:
            df = df.cumsum()
        if not threshold:
            if not specific_day:
                res = df[grouping].idxmax() if how == "day" else df[grouping].max()
            else:
                if how == "day":
                    raise NotImplementedError(
                        "you specified the day, so you shouldn't be needing it as an answer"
                    )
                elif how == "amount":
                    res = df.loc[specific_day, grouping]
        else:
            if how == "amount":
                raise NotImplementedError(
                    "you specified the amount, so you shouldn't be needing it as an answer"
                )
            df = df[df[grouping] >= threshold][grouping]
            res = df.index[0] if len(df.index) > 0 else np.inf
        return {"value": res, "metric": metric_name}

    def group_count(self, batch: Batch, **params) -> List[List[str]]:
        l = []
        for df in batch.summed_output_list:
            if self.variants:
                l += [
                    self.count(gdf, **params) | {"variant": g}
                    for g, gdf in df.groupby("variant")
                ]
            else:
                l.append(self.count(df, **params))
        return pd.DataFrame(l)

    def sum_groupings(self, df: pd.DataFrame, how: str) -> pd.DataFrame:
        # TODO: this function needs to be cleaned up
        # this adds a column for analysis purposes, but duplicates data
        df[how] = df.filter(**self.states.get_filter(how, df.columns)).sum(
            axis="columns"
        )
        return df

    def r_0(self, batch: Batch) -> pd.DataFrame:
        df_list = batch.summed_list
        res = [
            (df["daily_infected"] / (df["daily_infectors"]))
            .replace([np.inf, -np.inf], 0)
            .mean()
            for df in df_list
        ]
        return pd.DataFrame({"value": res, "metric": ["r_0"] * len(res)})


class SensitivityRunner(ConnectedBasicBlock):  # (Runner?)
    def _times(self, sr: dict) -> int:
        return int((sr["max"] - sr["min"]) / sr["step"])

    def _iter_values(
        self,
        sr: Dict[str, float | int],
        baseline: List[float | int] | float | int,
    ) -> List[float | int]:
        range_ = [
            round(x, 4)
            for x in np.linspace(sr["min"], sr["max"], num=self._times(sr) + 1).tolist()
        ]
        change = self.task["sensitivity"]["change"]
        if isinstance(baseline, list):
            if change in ("first", "second"):
                const = [baseline[(1 if "first" else 0)]] * len(range_)
                if change == "first":
                    return list(zip(range_, const))
                else:
                    return list(zip(const, range_))
            elif change == "both":
                return list(product(range_, range_))
            else:
                raise
        else:
            return range_

    def run(self) -> MultiBatch:
        cr = ContagionRunner(self.dataset, self.task)
        multibatch = MultiBatch(
            self.dataset, self.task, Analysis(self.dataset, self.task)
        )
        sa_conf = self.task["sensitivity"]
        for param in sa_conf["params"]:
            print(f"running sensitivity analysis on {param}")
            if param not in self.task.keys():
                sub = [k for k in self.task["paths"][param].keys() if k[0] == "d"][0]
                baseline = self.task["paths"][param][sub]
                sr = sa_conf["ranges"][param]
                times = self._times(sr)
                for i in range(times + 1):
                    v = round(sr["min"] + i * sr["step"], 4)
                    value = (
                        [v, baseline[1]] if sub == "duration" else [v, round(1 - v, 4)]
                    )
                    print(f"checking when {param} {sub} = {value}")
                    self.task["paths"][param].update({sub: value})
                    batch = cr.run()
                    multibatch.append_batch(
                        batch=batch, param=f"{param}__{sub}", step=v
                    )
                self.task["paths"][param][sub] = baseline
            else:
                baseline = self.task[param]
                sr = sa_conf["ranges"][param]
                times = self._times(sr)
                iv = self._iter_values(sr, baseline)
                for value in iv:
                    print(f"checking when {param} = {value}")
                    self.task.update({param: value})
                    batch = cr.run()
                    multibatch.append_batch(batch=batch, param=param, step=value)
                self.task[param] = baseline
        return multibatch
