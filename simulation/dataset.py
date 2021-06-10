from datetime import date, datetime

import numpy as np
import pandas as pd
from yaml import Loader, load

from simulation.constants import *
from simulation.google_cloud import GoogleCloud
from simulation.task import Task


class Dataset(object):
    def __init__(self, name, task=Task):
        self.name = name
        self.task = task
        with open(CONFIG_FOLDER / "datasets.yaml", "r") as f:
            metadata = load(f, Loader=Loader)[name]
        for k, v in metadata.items():
            if k in ["start_date", "end_date"]:
                setattr(self, k, self._strp(metadata[k]))
            else:
                setattr(self, k, v)
        if hasattr(self, "start_date"):
            self.period: int = (self.end_date - self.start_date).days

    def _strp(self, d: str) -> date:
        return datetime.strptime(d, "%Y-%m-%d").date()

    def load_dataset(self, gcloud: GoogleCloud):
        if self.storage == "csv":
            if not (DATA_FOLDER / f"{self.name}.csv").exists():
                gcloud.download(f"{self.name}.csv")
            self.data = pd.read_csv(
                DATA_FOLDER / f"{self.name}.csv", parse_dates=["datetime"]
            )
        self.data = self.data.drop(
            columns=set(self.data.columns)
            - {"group", "source", "destination", "datetime", "duration", "hops"}
        )
        if self.groups:
            self.data["group"] = self.data["group"].apply(eval)
        if not self.data["datetime"].dtype == "<M8[ns]":
            raise ValueError("Can't split to days without actual timestamps")
        if "duration" not in self.data.columns:
            self.data["duration"] = 5
        self._split()
        self._get_ids()
        if not hasattr(self, "start_date"):
            self.start_date = self.split[0]["datetime"].min().date()
            self.end_date = self.split[max(self.split)]["datetime"].max().date()
            self.period: int = max(self.split)
        if self.name == "milan_calls":
            self._load_helper_dfs(gcloud)

    def _get_ids(self):
        if not self.groups:
            self.ids = {
                x: list(df[["source", "destination"]].stack().drop_duplicates())
                for x, df in self.split.items()
            }
        else:
            self.ids = {
                x: list(set.union(*df["group"])) for x, df in self.split.items()
            }
        if not hasattr(self, "nodes"):
            self.nodes = len(set.union(*[set(i) for i in self.ids.values()]))

    def _split(self):
        if not self.real:
            self.split = {
                i: x[1]
                for i, x in enumerate(
                    self.data.resample(
                        f"{5*self.task['window_size']}min", on="datetime"
                    )
                )
            }
        else:
            samplesize = f"{self.task['squeeze']}D" if self.task["squeeze"] > 1 else "D"
            self.split = {
                i: x[1]
                for i, x in enumerate(self.data.resample(samplesize, on="datetime"))
            }
            if self.task["squeeze"] < 1:
                d, i = {}, 0
                for df in self.split.values():
                    for df_frac in np.array_split(
                        df.sample(frac=1), round(self.task["squeeze"] ** -1)
                    ):
                        d[i] = df_frac
                        i += 1
                self.split = d

    def _load_helper_dfs(self, gcloud):
        gcloud.download(f"{self.name}_demography.feather")
        self.demography = pd.read_feather(
            DATA_FOLDER / f"{self.name}_demography.feather"
        )
        gcloud.download(f"{self.name}_zero.feather")
        self.zeroes = pd.read_feather(DATA_FOLDER / f"{self.name}_zero.feather")
