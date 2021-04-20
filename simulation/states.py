from yaml import load, Loader

from simulation.constants import CONFIG_FOLDER


class States:
    def __init__(self):
        with open(CONFIG_FOLDER / "config.yaml") as f:
            config = load(f, Loader=Loader)
        self.states = {k for k in config["paths"].keys()}
        self.non_states = {
            "infected",
            "infectors",
            "infected_daily",
            "daily_infectors",
            "sick",
        }
        self.aggregate_states = {"stable", "intensive_care", "red", "purple"}
        self.sick_states = self.states - {"green", "white", "black"} - self.non_states
        self.states_with_duration = self.states - self.aggregate_states
        self.all = sorted(self.non_states | self.aggregate_states | self.states)

    @staticmethod
    def decrypt_states(color: str) -> str:
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
