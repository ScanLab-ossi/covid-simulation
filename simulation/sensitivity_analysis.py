from __future__ import annotations
from simulation.helpers import timing

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd

from simulation.building_blocks import BasicBlock, ConnectedBasicBlock
from simulation.constants import *
from simulation.contagion import ContagionRunner
from simulation.output import MultiBatch

if TYPE_CHECKING:
    from simulation.output import Batch


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
        threshold: Optional[Union[int, float]] = None,
        percent: bool = True,
        how: str = "amount",
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
        if grouping not in df.columns:
            df = df.pipe(self.sum_groupings, how=grouping)
        if percent:
            df = df / self.dataset.nodes
        if cumsum:
            df = df.cumsum()
        if not threshold:
            res = df[grouping].idxmax() if how == "day" else df[grouping].max()
        else:
            if how == "amount":
                raise NotImplementedError(
                    "you specified the amount, so you shouldn't be needing it as an answer"
                )
            df = df[df[grouping] >= threshold][grouping]
            res = df.index[0] if len(df.index) > 0 else np.inf
        return [res, metric_name]

    def group_count(self, batch: Batch, **params) -> List[List[str]]:
        l = []
        for df in batch.summed_output_list:
            l.append(self.count(df, **params))
        return l

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

    def _steps(self, value: float, baseline: float, step: float) -> str:
        step = int(round((value - baseline) / step, 1))
        return f"{('+' if step > 0 else '')}{step}"

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
                    v = round(sr["min"] + i * sr["step"], 2)
                    value = (
                        [v, baseline[1]] if sub == "duration" else [v, round(1 - v, 2)]
                    )
                    print(f"checking when {param} {sub} = {value}")
                    self.task["paths"][param].update({sub: value})
                    batch = cr.run()
                    multibatch.append_batch(
                        batch=batch, param=f"{param}__{sub}", step=v
                    )  # self._steps(v, baseline[0], sr["step"])
                self.task["paths"][param][sub] = baseline
            else:
                baseline = self.task[param]
                sr = sa_conf["ranges"][param]
                times = self._times(sr)
                for i in range(times + 1):
                    value = round(sr["min"] + i * sr["step"], 2)  # wierd float stuff
                    print(f"checking when {param} = {value}")
                    self.task.update({param: value})
                    batch = cr.run()
                    multibatch.append_batch(batch=batch, param=param, step=value)
                self.task[param] = baseline
        return multibatch
