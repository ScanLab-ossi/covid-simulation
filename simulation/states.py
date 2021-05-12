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
            "daily_infected",
            "daily_infectors",
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
            return f"{color} ({d[color]})"
        except KeyError:
            return color
