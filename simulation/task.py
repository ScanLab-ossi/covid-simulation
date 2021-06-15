from collections import UserDict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from google.cloud.datastore import Entity  # type: ignore
from yaml import Loader, load

from simulation.constants import *


class Task(UserDict):
    """
    Config for each simulation run.
    """

    def __init__(
        self,
        data: dict = {},
        done: bool = False,
        path: Path = CONFIG_FOLDER / "config.yaml",
    ):
        super().__init__(dict(data))
        self.id: int = (
            data.id if isinstance(data, Entity) else np.random.randint(1e15, 1e16)
        )
        self.path = path
        self.setdefault("done", done)
        self.setdefault("start_date", datetime.now())
        self.load_config()
        if self.get("country"):
            self.load_country_info()
        # if settings["LOCAL"]:
        #     self.load_state_transition()

    def load_config(self):
        with open(self.path) as f:
            config = load(f, Loader=Loader)
        for k, v in {
            **config["meta"],
            **config["params"],
            "sensitivity": config["sensitivity"],
            "paths": config["paths"],
        }.items():
            self.setdefault(k, v)

    def load_country_info(self):
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

    def variants(self) -> List[str]:
        return [k for k in self.keys() if k.startswith("variant_")]
