from yaml import load, Loader
from simulation.constants import CONFIG_FOLDER


class Metrics:
    def __init__(self):
        self.not_colors = {"infected_daily", "daily_infectors"}
        self.aggregate_colors = {"stable", "intensive_care", "red", "purple"}
        with open(CONFIG_FOLDER / "config.yaml") as f:
            config = load(f, Loader=Loader)
        self.states = {k for k in config["paths"].keys()}
        self.states_with_duration = self.states - self.aggregate_colors
        self.all = sorted(self.not_colors | self.aggregate_colors | self.states)

    @staticmethod
    def decrypt_colors(color: str, replace=False) -> str:
        d = {
            "blue": "asymptomatic",
            "pink": "mild symptoms",
            "purple": "pre-symptomatic",
            "red": "severe symptoms, hospitalized",
            "white": "recovered",
            "black": "deceased",
            "green": "susceptible",
        }
        try:
            return f"{color} ({d[color]})"
        except KeyError:
            return color
