from math import comb, floor
from typing import List, Literal, Union

import pandas as pd
from numpy import random
import numpy as np
from yaml import Loader, load

from simulation.constants import *
from simulation.google_cloud import GoogleCloud
from simulation.helpers import timing
from simulation.task import Task


class Dataset(object):
    def __init__(
        self, name, task: Task, gcloud: GoogleCloud, reproducible: bool = False
    ):
        self.name = name
        self.task = task
        self.gcloud = gcloud
        self.rng = random.default_rng(42 if reproducible else None)
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
        filename = f"{self.name}.csv"
        if self.storage == "csv":
            if not (DATA_FOLDER / filename).exists():
                print("here")
                self.gcloud.download(filename)
            data = pd.read_csv(
                DATA_FOLDER / filename,
                parse_dates=["datetime"],
                dtype={"source": int, "destination": int},
            )
        data = self._prep(data)
        if cols := self.task.get("randomize"):
            data = self._randomize(data, cols)
        self._split(data=data)
        self._remove_redundant_days()
        self._get_ids()
        if self.name == "milan_calls":
            self._load_helper_dfs()

    def _prep(self, data: pd.DataFrame) -> pd.DataFrame:
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
        return data

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
                    for x in self.task["visualize"]["metrics"]
                    if "specific_day" in x
                ][0]
                self.split = {k: v for k, v in self.split.items() if k <= max_day}
            except IndexError:
                pass

    def _split(self, data: pd.DataFrame):
        samplesize = f"{floor(5 * self.task['tau'] / self.task['divide'])}min"
        self.split = {
            i: x[1] for i, x in enumerate(data.resample(samplesize, on="datetime"))
        }
        # t = [
        #     np.array_split(day, self.task["divide"])
        #     for _, day in data.resample("1d", on="datetime")
        # ]
        # self.split = {
        #     i: x for i, x in enumerate([item for sublist in t for item in sublist])
        # }

    def _load_helper_dfs(self):
        self.gcloud.download(f"{self.name}_demography.feather")
        self.demography = pd.read_feather(
            DATA_FOLDER / f"{self.name}_demography.feather"
        )
        self.gcloud.download(f"{self.name}_zero.feather")
        self.zeroes = pd.read_feather(DATA_FOLDER / f"{self.name}_zero.feather")

    def _randomize(
        self, data: pd.DataFrame, cols: List[Union[str, List[str]]]
    ) -> pd.DataFrame:
        for col in cols:
            data[col] = random.permutation(data[col].to_numpy())
        data = data.sort_values(by="datetime").reset_index(drop=True)
        return data

    def temporal_density(
        self, how: Literal["mean", "list"] = "mean"
    ) -> float | List[float]:
        l = []
        for df in self.split.values():
            for _, timeframe in df.resample("5min", on="datetime"):
                l.append(len(timeframe) / comb(self.nodes, 2))
        if how == "mean":
            l = sum(l) / len(l)
        return l
