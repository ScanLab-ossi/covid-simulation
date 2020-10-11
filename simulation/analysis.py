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
        grouping: str = "sick",  # a color, sick or infected
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
        metric_name = f"{'max' if max_ and percent==None and amount==None else 'day_of_specific'}_{'percent'if percent else 'amount'}_{grouping}"
        if avg:
            return sum(res) / len(res)
        else:
            return pd.DataFrame({"value": res, "metric": [metric_name] * len(res)})

    def sum_groupings(
        self, df_list: List[pd.DataFrame], how: str
    ) -> List[pd.DataFrame]:
        if how == "red":
            filter_ = {"regex": r"(intensive|stable)\w+"}
        elif how == "sick":
            filter_ = {"items": set(df_list[0].columns) - {"green", "white", "black"}}
        else:
            filter_ = {"like": how}
        for df in df_list:
            df[how] = df.filter(**filter_).sum(axis="columns")
        return df_list

    def r_0(self, batch: Batch, what: str = "total") -> pd.DataFrame:
        fname = "r_0"
        if what == "total":
            sick = self.sick(batch, "total")
            infectors = np.array(
                [len(set().union(*output.df["infector"].dropna())) for output in batch]
            )
            return pd.DataFrame(sick["sick"].values / infectors, columns=[fname])
        elif what == "average":
            return (
                pd.concat([self.basic_r_0(output.df) for output in batch])
                .groupby("infection_date")[["r_0", "r_thresh"]]
                .mean()
                .reset_index()
            )

    def basic_r_0(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.groupby("infection_date").agg(
            {
                "infector": lambda x: len(set().union(*x.dropna())),
                "final_state": "count",
            }
        )
        df = df.assign(r_0=df["final_state"] / df["infector"])
        df.index = df.index.astype("datetime64[ns]", copy=False)
        idx = pd.date_range(self.dataset.start_date, self.dataset.end_date)
        df = (
            df.reindex(idx, fill_value=0)
            .rename_axis("infection_date", axis="index")
            .reset_index()
        )
        df["r_thresh"] = 1
        return df
