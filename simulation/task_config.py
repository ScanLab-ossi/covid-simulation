from pathlib import Path
import json, subprocess
import numpy as np
from collections import UserDict
from datetime import datetime

from simulation.constants import *


class TaskConfig(UserDict):
    """
    Config for each simulation run. Can be used for validation
    """

    def __init__(self, data, done=False):
        super().__init__(dict(data))
        self.metadata_keys = [
            "dataset",
            "repetitions",
            "start_date",
            "end_date",
            "output_url",
            "machine_version",
            "done",
        ]
        self.continuous_params = [
            "number_of_patient_zero",
            "alpha_blue",
            "D_min",
            "D_max",
            "P_max",
            "threshold",
        ]
        self.distribution_params = [
            "age_dist",
            "blue_to_white",
            "purple_to_red",
            "red_to_final_state",
            "P_r",
        ]
        self.ordinal_params = ["infection_model"]
        # self.params = { k: v
        #     for k, v in data.items()
        #     if k not in self.metadata_keys + self.run_configuration_keys
        # }
        self.data["machine_version"] = self.get_machine_version()
        self.data.setdefault("start_date", datetime.now())
        self.data.setdefault("repetitions", REPETITIONS)
        self.data.setdefault("dataset", DATASET)
        self.data.setdefault("done", done)

    def get_params(self):
        return {k: v for k, v in self.data.items() if k not in self.metadata_keys}

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
