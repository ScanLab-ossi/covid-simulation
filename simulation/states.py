from typing import Optional, List

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
        }
        self.aggregate_states = {"stable", "intensive_care", "red", "purple"}
        self.sick_states = self.states - {"green", "white", "black"} - self.non_states
        self.states_with_duration = self.states - self.aggregate_states
        self.all = sorted(self.non_states | self.aggregate_states | self.states)

    def decrypt_states(self, color: str) -> str:
        d = {
            "blue": "asymptomatic",
            "pink": "mild symptoms, quarantined",
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
        try:
            return d[color]
        except KeyError:
            return color

    def get_filter(self, how: str, l: Optional[List[str]] = None):
        if how == "red":
            return {"regex": r"(intensive|stable)\w+"}
        elif how == "sick":
            return {"items": set(l) - self.sick_states}
        elif how == "not_green":
            return {"items": set(l) - {"green"} - self.non_states}
        else:
            return {"like": how}
