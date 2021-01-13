import pandas as pd
import numpy as np
from typing import Optional, List, Union

from simulation.output import Batch, MultiBatch
from simulation.building_blocks import BasicBlock
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.constants import *


class Analysis(BasicBlock):
    def __init__(
        self, dataset: Dataset, task: Task, df: Optional[pd.DataFrame] = None,
    ):
        super().__init__(dataset=dataset, task=task)
        self.got_input = isinstance(df, pd.DataFrame)

    def count(
        self,
        batch: Batch,
        grouping: str = "sick",  # a color, sick, infected
        percent: int = None,
        amount: int = None,
        max_: bool = True,
        how: str = "day",  # amount
        avg: bool = True,
    ) -> Union[int, pd.DataFrame]:
        # amount / day @ max percent / amount of color / sick
        # day @ specific percent / amount of color / sick
        df_list = batch.summed_list
        if grouping not in df_list[0].columns:
            df_list = self.sum_groupings(df_list, grouping)
        if percent:
            df_list = [df * 100 / self.dataset.nodes for df in df_list]
        if max_:
            # for df in df_list:
            #     print(df[grouping])
            res = [
                df[grouping].idxmax() if how == "day" else df[grouping].max()
                for df in df_list
            ]
        else:
            if how == "amount":
                raise NotImplementedError(
                    "you specified the amount, so you shouldn't be needing it as an answer"
                )
            thresh_dfs = [
                df[df[grouping] >= (percent if percent else amount)][grouping]
                for df in df_list
            ]
            res = [(df.index[0] if len(df.index) > 0 else np.inf) for df in thresh_dfs]
        metric_name = f"{'max' if max_ and percent==None and amount==None else 'day_of_specific'}_{'percent'if percent else how}_{grouping}"
        if avg:
            return sum(res) / len(res)
        else:
            return pd.DataFrame({"value": res, "metric": [metric_name] * len(res)})

    def sum_groupings(
        self, df_list: List[pd.DataFrame], how: str
    ) -> List[pd.DataFrame]:
        # TODO: this function needs to be cleaned up
        non_states = {
            "infected",
            "infectors",
            "infected_daily",
            "daily_infectors",
            "sick",
        }
        if how == "red":
            filter_ = {"regex": r"(intensive|stable)\w+"}
        elif how == "sick":
            filter_ = {
                "items": set(df_list[0].columns)
                - {"green", "white", "black"}
                - non_states
            }
        elif how == "not_green":
            filter_ = {"items": set(df_list[0].columns) - {"green"} - non_states}
        else:
            filter_ = {"like": how}
        for df in df_list:
            # this adds a column for analysis purposes, but duplicates data
            df[how] = df.filter(**filter_).sum(axis="columns")
        return df_list

    def r_0(self, batch: Batch) -> pd.DataFrame:
        df_list = batch.summed_list
        res = [
            (df["infected_daily"] / (df["daily_infectors"]))
            .replace([np.inf, -np.inf], 0)
            .mean()
            for df in df_list
        ]
        return pd.DataFrame({"value": res, "metric": ["r_0"] * len(res)})
