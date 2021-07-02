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
            setattr(self, k, v)

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
        self._remove_redundant_days()
        self._get_ids()
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

    def _remove_redundant_days(self):
        if self.task["SENSITIVITY"]:
            try:
                max_day = [
                    x["specific_day"]
                    for x in self.task["sensitivity"]["metrics"]
                    if "specific_day" in x
                ][0]
                self.split = {k: v for k, v in self.split.items() if k <= max_day}
            except KeyError:
                pass

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
