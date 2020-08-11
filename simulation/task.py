from pathlib import Path
import json, subprocess
import numpy as np  # type: ignore
from collections import UserDict
from datetime import datetime
from google.cloud.datastore import Entity  # type: ignore
from yaml import load, Loader
from typing import Union

from simulation.constants import *


class Task(UserDict):
    """
    Config for each simulation run.
    """

    def __init__(self, data: dict = {}, done: bool = False):
        with open(CONFIG_FOLDER / "config.yaml") as f:
            config = load(f, Loader=Loader)
        super().__init__(dict(data))
        self.id: int = data.id if isinstance(data, Entity) else np.random.randint(
            1e15, 1e16
        )
        for k, v in {
            "start_date": datetime.now(),
            "done": done,
            **config["meta"],
            **config["params"],
            "sensitivity": config["sensitivity"],
        }.items():
            self.data.setdefault(k, v)
