from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal

import numpy as np
import pandas as pd

from simulation.building_blocks import BasicBlock

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
        grouping="daily_infected", how="day"
        grouping="daily_infected", how="amount"
        grouping="sick", how="day"
        grouping="sick", how="amount"

        Raises
        ------
        NotImplementedError
            If threshold is set but how is specified.
        """
        # amount / day @ max percent / amount of color / sick
        # day @ specific percent / amount of color / sick
        metric_name = f"{'max' if not threshold else 'day_of_specific'}_{'percent' if not how else how}_{grouping}"
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
