from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

from simulation.building_blocks import BasicBlock
from simulation.constants import *

if TYPE_CHECKING:
    from simulation.output import Batch


class Analysis(BasicBlock):
    def count(
        self,
        df: pd.DataFrame,
        grouping: str = "sick",  # a color, , infected
        percent: int = None,
        amount: int = None,
        max_: bool = True,
        how: str = "day",  # amount
    ) -> List[str]:
        # amount / day @ max percent / amount of color / sick
        # day @ specific percent / amount of color / sick
        if grouping not in df.columns:
            df = df.pipe(self.sum_groupings, how=grouping)
        if percent:
            df = df * 100 / self.dataset.nodes
        if max_:
            res = df[grouping].idxmax() if how == "day" else df[grouping].max()
        else:
            if how == "amount":
                raise NotImplementedError(
                    "you specified the amount, so you shouldn't be needing it as an answer"
                )
            df = df[df[grouping] >= (percent if percent else amount)][grouping]
            res = df.index[0] if len(df.index) > 0 else np.inf
        metric_name = f"{'max' if max_ and percent==None and amount==None else 'day_of_specific'}_{'percent'if percent else how}_{grouping}"
        return [res, metric_name]

    def group_count(self, batch: Batch, **params) -> List[List[str]]:
        return [self.count(df, **params) for df in batch.summed_output_list]

    def sum_groupings(self, df: pd.DataFrame, how: str) -> pd.DataFrame:
        # TODO: this function needs to be cleaned up
        if how == "red":
            filter_ = {"regex": r"(intensive|stable)\w+"}
        elif how == "sick":
            filter_ = {"items": set(df.columns) - self.states.sick_states}
        elif how == "not_green":
            filter_ = {"items": set(df.columns) - {"green"} - self.states.non_states}
        else:
            filter_ = {"like": how}
        # this adds a column for analysis purposes, but duplicates data
        df[how] = df.filter(**filter_).sum(axis="columns")
        return df

    def r_0(self, batch: Batch) -> pd.DataFrame:
        df_list = batch.summed_list
        res = [
            (df["infected_daily"] / (df["daily_infectors"]))
            .replace([np.inf, -np.inf], 0)
            .mean()
            for df in df_list
        ]
        return pd.DataFrame({"value": res, "metric": ["r_0"] * len(res)})
