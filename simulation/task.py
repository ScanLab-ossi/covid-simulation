from collections import UserDict
from pathlib import Path
from typing import Literal, NoReturn, Optional

import numpy as np
import pandas as pd
from google.cloud.datastore import Entity  # type: ignore
from yaml import Loader, load, dump

from simulation.constants import *


class Task(UserDict):
    """
    Config for each simulation run.
    """

    def __init__(
        self,
        data: dict = {},
        path: Path = CONFIG_FOLDER / "config.yaml",
    ):
        super().__init__(dict(data))
        if isinstance(data, Entity) or "id" in data:
            self.id = data.id
        else:
            self.id = np.random.randint(1e15, 1e16)
        self.load_config(CONFIG_FOLDER / "default_config.yaml")
        self.load_config(path)
        self.path = OUTPUT_FOLDER / f"{self.id}.yaml"
        if self.get("country"):
            self.load_country_info()
        # if settings["LOCAL"]:
        #     self.load_state_transition()

    def load_config(self, path: Path):
        with open(path) as f:
            config = load(f, Loader=Loader)
            if not "default" in path.name:
                self.poi = config
        for key in ("meta", "params", "sensitivity", "paths"):
            for k, v in config.get(key, {}).items():
                if key in ("sensitivity", "paths"):
                    self.setdefault(key, {})
                    self[key][k] = v
                else:
                    self[k] = v

    def load_country_info(self) -> NoReturn:
        self["age_dist"] = (
            pd.read_csv(CONFIG_FOLDER / "population.csv")
            .set_index("Age Group")[self["country"]]
            .to_dict()
        )

    def load_state_transition(self, df: Optional[pd.DataFrame] = None):
        if not isinstance(df, pd.DataFrame):
            df = pd.read_csv(CONFIG_FOLDER / "state_transition.csv")
        df = df.set_index("state").T
        new_paths = pd.DataFrame([], columns=[], index=df.index)
        for k in self["paths"].keys():
            df_ = df.filter(like=f"d_{k}")
            if not df_.empty:
                new_paths[k] = df_.apply(list, axis=1)
        ss = {"green": "s1", "red": "s2", "stable": "s3", "intensive_care": "s4"}
        for k, v in ss.items():
            new_paths[k] = df[v].apply(lambda x: [1 - x, x])
        df["s1_p"] = (df["s1_p"] * 100).astype(int) / ((1 - df["s1"]) * 100).astype(int)
        df["s1_r"] = (df["s1_r"] * 100).astype(int) / ((1 - df["s1"]) * 100).astype(int)
        new_paths["purple"] = df[["s1_p", "s1_r"]].apply(list, axis=1)
        for k, v in new_paths.to_dict().items():
            if "distribution" in self["paths"][k]:
                self["paths"][k]["distribution"] = v
            elif "duration" in self["paths"][k]:
                self["paths"][k]["duration"] = v

    def export(
        self, what: Optional[str] = None, how: Literal["file", "print"] = "file"
    ) -> NoReturn:
        to_dump = getattr(self, what) if what else self.data
        if how == "file":
            with open(self.path, "w") as f:
                dump(to_dump, f, default_flow_style=False, sort_keys=False)
        elif how == "print":
            print(dump(to_dump, default_flow_style=False, sort_keys=False))
