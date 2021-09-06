from typing import List, Optional

import pandas as pd
from pandas.api.types import CategoricalDtype

from task import Task


class States:
    def __init__(self, task: Task):
        self.states = {k for k in task["paths"].keys()}
        self.non_states = {
            "infected",
            "infectors",
            "daily_infected",
            "daily_infectors",
            "daily_blue",
            "sick",
            "variant",
        }
        self.infectious_states = {"green", "purple_red", "purple_pink", "blue"}
        self.non_infectious_states = (
            self.states - self.infectious_states - self.non_states
        )
        self.aggregate_states = {"stable", "intensive_care", "red", "purple"}
        self.sick_states = self.states - {"green", "white", "black"} - self.non_states
        self.states_with_duration = self.states - self.aggregate_states
        self.all = sorted(self.non_states | self.aggregate_states | self.states)
        self.color_to_state = {
            "blue": "asymptomatic",
            "pink": "mild symptoms, quarantined",
            "purple": "presymptomatic",
            "purple_pink": "pre mild symptoms",
            "purple_red": "pre severe symptoms",
            "stable_black": "stable, before relapse",
            "stable_white": "stable, before recovery",
            "intensive_care_white": "ICU, before recovery",
            "intensive_care_white": "ICU, before relapse",
            "red": "severe symptoms, hospitalized",
            "white": "recovered",
            "black": "deceased",
            "green": "susceptible",
        }

    def categories(self, attr: str) -> CategoricalDtype:
        return CategoricalDtype(categories=getattr(self, attr), ordered=False)

    def categorify(self, df: pd.DataFrame, states: str = "states") -> pd.DataFrame:
        df["color"] = df["color"].astype(self.categories(states))
        return df

    def get_filter(self, how: str, l: Optional[List[str]] = None):
        if how == "red":
            return {"regex": r"(intensive|stable)\w+"}
        elif how == "sick":
            return {"items": set(l) - self.sick_states}
        elif how == "not_green":
            return {"items": set(l) - {"green"} - self.non_states}
        else:
            return {"like": how}


class Variants:
    def __init__(self, task: Task):
        self.list = [k for k in task.keys() if k.startswith("variant_")]
        self.exist = len(self.list) > 1
        self.categories = CategoricalDtype(categories=self.list, ordered=True)
        self.js = {
            variant: f"{variant}, j = {task[variant]['j']}" for variant in self.list
        }

    def categorify(self, df: pd.DataFrame) -> pd.DataFrame:
        df["variant"] = df["variant"].astype(self.categories)
        return df
