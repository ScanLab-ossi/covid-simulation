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
from simulation.analysis import Analysis

if TYPE_CHECKING:
    from output import Batch


class SensitivityRunner(ConnectedBasicBlock):
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
            if change in ("rows", "columns"):
                const = [baseline[(1 if "first" else 0)]] * len(range_)
                if change == "rows":
                    return list(zip(range_, const))
                else:
                    return list(zip(const, range_))
            elif change == "matrix":
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
