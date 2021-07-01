from datetime import date, datetime
from math import floor

import pandas as pd
from yaml import Loader, load

from simulation.constants import *
from simulation.google_cloud import GoogleCloud
from simulation.task import Task
from simulation.helpers import timing


class Dataset(object):
    def __init__(self, name, task=Task, gcloud=GoogleCloud):
        self.name = name
        self.task = task
        self.gcloud = gcloud
        with open(CONFIG_FOLDER / "datasets.yaml", "r") as f:
            datasets = load(f, Loader=Loader)
            try:
                metadata = datasets[name]
            except KeyError:
                metadata = datasets[[k for k in datasets.keys() if k in name][0]]
        for k, v in metadata.items():
            if k in ["start_date", "end_date"]:
                setattr(self, k, self._strp(metadata[k]))
            else:
                setattr(self, k, v)
        if hasattr(self, "start_date"):
            self.period: int = (self.end_date - self.start_date).days

    def _strp(self, d: str) -> date:
        return datetime.strptime(d, "%Y-%m-%d").date()

    @timing
    def load_dataset(self):
        if self.storage == "csv":
            if not (DATA_FOLDER / f"{self.name}.csv").exists():
                self.gcloud.download(f"{self.name}.csv")
            data = pd.read_csv(
                DATA_FOLDER / f"{self.name}.csv", parse_dates=["datetime"]
            )
        data = data.drop(
            columns=set(data.columns)
            - {"group", "source", "destination", "datetime", "duration", "hops"}
        )
        if self.groups:
            data["group"] = data["group"].apply(eval)
        if not data["datetime"].dtype == "<M8[ns]":
            raise ValueError("Can't split to days without actual timestamps")
        if "duration" not in data.columns:
            data["duration"] = 5
        self._split(data=data)
        self._get_ids()
        if not hasattr(self, "start_date"):
            self.start_date = self.split[0]["datetime"].min().date()
            self.end_date = self.split[max(self.split)]["datetime"].max().date()
            self.period: int = max(self.split)
        if self.name == "milan_calls":
            self._load_helper_dfs()

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

    def _split(self, data: pd.DataFrame):
        samplesize = f"{floor(5 * self.task['window_size'] / self.task['divide'])}min"
        self.split = {
            i: x[1] for i, x in enumerate(data.resample(samplesize, on="datetime"))
        }

    def _load_helper_dfs(self):
        self.gcloud.download(f"{self.name}_demography.feather")
        self.demography = pd.read_feather(
            DATA_FOLDER / f"{self.name}_demography.feather"
        )
        self.gcloud.download(f"{self.name}_zero.feather")
        self.zeroes = pd.read_feather(DATA_FOLDER / f"{self.name}_zero.feather")
