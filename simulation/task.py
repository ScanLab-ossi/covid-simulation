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
        # self.data["machine_version"] = self.get_machine_version()
        for k, v in {
            "start_date": datetime.now(),
            "done": done,
            **config["meta"],
            **config["params"],
            "sensitivity": config["sensitivity"],
        }.items():
            self.data.setdefault(k, v)

    def get_machine_version(self):
        versions = []
        for f in ["contagion", "state_transition"]:
            versions.append(
                subprocess.check_output(
                    [f'git log -1 --pretty="%h" ./simulation/{f}.py'], shell=True
                )
                .strip()
                .decode("utf-8")
            )
        newer = (
            subprocess.check_output(
                " ".join(["git merge-base --is-ancestor"] + versions + ["; echo $?"]),
                shell=True,
            )
            .strip()
            .decode("utf-8")
        )  # 0 = contagion is newer, 1 = st is newer
        return versions[int(newer)]
