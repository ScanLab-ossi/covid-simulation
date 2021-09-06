from typing import Union, Dict, List

import numpy as np
import pandas as pd

from helpers import timing
from building_blocks import RandomBasicBlock
from constants import *


class StateTransition(RandomBasicBlock):
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

    def move_one(self, row: pd.Series) -> pd.Series:
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
