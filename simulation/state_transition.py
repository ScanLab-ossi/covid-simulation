from datetime import timedelta, date
from typing import Tuple, Union

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
        self.paths = task.paths
        self.rng = np.random.default_rng()

    def move_one(self, row: pd.Series, day: int) -> pd.Series:
        if row["infection_date"] > day:
            return row
        if row["days_left"] > 0:
            row["days_left"] -= 1
            return row
        else:
            while row["days_left"] == 0:
                d = self.paths[row["color"]]
                try:
                    path_dist = d["distribution"] if len(d["children"]) > 1 else [1]
                    next_state = self.rng.choice(d["children"], 1, p=path_dist).item()
                    row["color"] = next_state
                    row["days_left"] = self.paths[next_state].get("duration", 0)
                    if row["days_left"] != 0:
                        norm = self.rng.normal(*row["days_left"])
                        row["days_left"] = int(np.maximum(np.around(norm), 1)) - 1
                except KeyError:
                    return row
            return row
