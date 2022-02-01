from collections import UserDict
from pathlib import Path
from typing import Literal

import numpy as np
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
        if "id" in data:
            self.id = data.id
        else:
            self.id = np.random.randint(1e15, 1e16)
        self.load_config(CONFIG_FOLDER / "default_config.yaml")
        self.load_config(path)
        self.path = OUTPUT_FOLDER / f"{self.id}.yaml"
        # if settings["LOCAL"]:
        #     self.load_state_transition()

    def load_config(self, path: Path):
        with open(path) as f:
            config = load(f, Loader=Loader)
            if not "default" in path.name:
                self.poi = config
        for key in ("meta", "params", "sensitivity", "paths", "output"):
            for k, v in config.get(key, {}).items():
                if key in ("sensitivity", "paths", "output"):
                    self.setdefault(key, {})
                    self[key][k] = v
                else:
                    self[k] = v

    def export(self, what: str | None = None, how: Literal["file", "print"] = "file"):
        to_dump = getattr(self, what) if what else self.data
        if how == "file":
            with open(self.path, "w") as f:
                dump(to_dump, f, default_flow_style=False, sort_keys=False)
        elif how == "print":
            print(dump(to_dump, default_flow_style=False, sort_keys=False))

    def get(self, attr, alt=False, variant=None):
        if attr not in self:
            return alt
        else:
            this = self[attr]
            if (
                variant
                and (isinstance(this, list) or isinstance(this, tuple))
                and len(this) == len(self["variants"])
            ):
                return this[self["variants"].index(variant)]
            else:
                return this
