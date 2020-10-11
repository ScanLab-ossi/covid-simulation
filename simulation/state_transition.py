from datetime import timedelta, date
from typing import Tuple, Union, Dict, List

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from simulation.dataset import Dataset
from simulation.task import Task
from simulation.output import Output
from simulation.helpers import timing
from simulation.building_blocks import BasicBlock
from simulation.constants import *


class StateTransition(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task):
        self.task = task
        self.dataset = dataset
        self.rng = np.random.default_rng()

    # can cache this?
    def _get_age(
        self,
        d: Dict[str, Union[List[str], Dict[str, float], List[float]]],
        key: str,
        age: int,
    ) -> List[float]:
        age = int(age.split("-")[0]) if isinstance(age, str) else age
        for k in d[key].keys():
            if age in range(*[int(x) for x in k.split("-")]):
                return d[key][k]

    def move_one(self, row: pd.Series, day: int) -> pd.Series:
        if row["days_left"] > 0:
            row["days_left"] -= 1
            return row
        else:
            while row["days_left"] == 0:
                d = self.task["paths"][row["color"]]
                try:
                    if len(d["children"]) == 1:
                        path_dist = [1]
                    elif isinstance(d["distribution"], dict):
                        path_dist = self._get_age(d, "distribution", row["age"])
                    else:
                        path_dist = d["distribution"]
                    next_state = self.rng.choice(d["children"], 1, p=path_dist).item()
                    row["color"] = next_state
                    duration = self.task["paths"][next_state].get("duration", 0)
                    if duration == 0:
                        row["days_left"] = 0
                    else:
                        if isinstance(duration, dict):
                            row["days_left"] = self._get_age(d, "duration", row["age"])
                        norm = self.rng.normal(*duration)
                        row["days_left"] = int(np.maximum(np.around(norm), 1)) - 1
                except KeyError:
                    return row
            return row
