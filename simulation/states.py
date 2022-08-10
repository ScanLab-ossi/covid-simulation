from typing import Dict, List

import pandas as pd
from pandas.api.types import CategoricalDtype

from simulation.task import Task


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
            "r_0",
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
            "blue": "infected",
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
        df["state"] = df["state"].astype(self.categories(states))
        return df

    def get_filter(self, how: str, l: List[str] | None = None):
        if how == "red":
            return {"regex": r"(intensive|stable)\w+"}
        elif how == "sick":
            return {"items": set(l) - self.sick_states}
        elif how == "not_green":
            return {"items": set(l) - {"green"} - self.non_states}
        else:
            return {"like": how}


class Variants(list):
    def __init__(self, task: Task):
        super().__init__(task.get("variants", alt=[]))
        self.task = task
        if self and task["infection_model"] != "VariantInfection":
            raise
        if self:
            self.categories = CategoricalDtype(categories=self, ordered=True)
        self.column = ["variant"] if self else []

    def variant_to_param(self, param: str) -> Dict[str, str]:
        return {k: self.task.get(param, variant=k) for k in self}

    def categorify(self, df: pd.DataFrame) -> pd.DataFrame:
        if self:
            df["variant"] = df["variant"].astype(self.categories)
        return df
