from datetime import date, datetime

import numpy as np
import pandas as pd
from yaml import Loader, load

from simulation.constants import *
from simulation.google_cloud import GoogleCloud
from simulation.task import Task


class Dataset(object):
    def __init__(self, name):
        self.name = name
        with open(CONFIG_FOLDER / "datasets.yaml", "r") as f:
            self.metadata = load(f, Loader=Loader)[name]
        self.storage: str = self.metadata["storage"]
        self.nodes: int = self.metadata["nodes"]
        self.start_date: date = self._strp(self.metadata["start_date"])
        self.end_date: date = self._strp(self.metadata["end_date"])
        self.interval: str = self.metadata["interval"]
        self.groups: bool = self.metadata["groups"]
        self.hops: bool = self.metadata["hops"]
        self.period: int = (self.end_date - self.start_date).days

    def _strp(self, d: str) -> date:
        return datetime.strptime(d, "%Y-%m-%d").date()

    def load_dataset(self, gcloud: GoogleCloud):
        if self.storage == "csv":
            if not (DATA_FOLDER / f"{self.name}.csv").exists():
                gcloud.download(f"{self.name}.csv")
            self.data = pd.read_csv(
                DATA_FOLDER / f"{self.name}.csv",
                parse_dates=["datetime"],
                usecols=(["group"] if self.groups else ["source", "destination"])
                + ["datetime", "duration"]
                + (["hops"] if self.hops else []),
            )
            task = Task()
            samplesize = f"{task['squeeze']}D" if task["squeeze"] > 1 else "D"
            self.split = {
                i: x[1]
                for i, x in enumerate(self.data.resample(samplesize, on="datetime"))
            }
            if self.groups:
                self.data["group"] = self.data["group"].apply(eval)
                self.ids = {
                    x: list(set.union(*df["group"])) for x, df in self.split.items()
                }
                if task["squeeze"] < 1:
                    d, i = {}, 0
                    for df in self.split.values():
                        for df_frac in np.array_split(
                            df.sample(frac=1), round(task["squeeze"] ** -1)
                        ):
                            d[i] = df_frac
                            i += 1
                    self.split = d
            else:
                if task["squeeze"] < 1:
                    raise ValueError(
                        "Can't unsqueeze dataset with group interactions in pairwise form"
                    )
                self.ids = {
                    x: list(df[["source", "destination"]].stack().drop_duplicates())
                    for x, df in self.split.items()
                }
        if self.name == "milan_calls":
            self.load_helper_dfs(gcloud)

    def load_helper_dfs(self, gcloud):
        gcloud.download(f"{self.name}_demography.feather")
        self.demography = pd.read_feather(
            DATA_FOLDER / f"{self.name}_demography.feather"
        )
        gcloud.download(f"{self.name}_zero.feather")
        self.zeroes = pd.read_feather(DATA_FOLDER / f"{self.name}_zero.feather")


# more elegant json reading
#  for key in self.datasets[name]:
#     if "date" in key:
#         setattr(
#             self,
#             key,
#         )
#     else:
#         setattr(self, key, self.datasets[name][key])
