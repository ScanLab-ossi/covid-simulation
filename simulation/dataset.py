from datetime import datetime, date
import pandas as pd
import numpy as np
import json  # type: ignore

from simulation.constants import *
from simulation.google_cloud import GoogleCloud
from simulation.helpers import timing


class Dataset(object):
    def __init__(self, name):
        self.name = name
        with open(CONFIG_FOLDER / "datasets.json", "r") as f:
            self.metadata = json.load(f)[name]
        self.storage: str = self.metadata["storage"]
        self.nodes: int = self.metadata["nodes"]
        self.start_date: date = self._strp(self.metadata["start_date"])
        self.end_date: date = self._strp(self.metadata["end_date"])
        self.interval: str = self.metadata["interval"]
        self.infection_param: str = self.metadata["infection_param"]
        self.groups: bool = self.metadata["groups"]
        self.hops: bool = self.metadata["hops"]
        self.period: int = (self.end_date - self.start_date).days

    def _strp(self, d: str) -> date:
        return datetime.strptime(d, "%Y-%m-%d").date()

    # @timing
    def load_dataset(self, gcloud: GoogleCloud):
        if self.storage == "csv":
            gcloud.download(f"{self.name}.csv")
        self.data = pd.read_csv(
            DATA_FOLDER / f"{self.name}.csv",
            parse_dates=["datetime"],
            usecols=(["group"] if self.groups else ["source", "destination"])
            + ["datetime", "duration"]
            + (["hops"] if self.hops else []),
        )
        if self.groups:
            self.data["group"] = self.data["group"].apply(eval)
        self.split = {x.date(): df for x, df in self.data.resample("D", on="datetime")}
        self.ids = {
            x: list(df[["source", "destination"]].stack().drop_duplicates())
            for x, df in self.split.items()
        }


# more elegant json reading
#  for key in self.datasets[name]:
#     if "date" in key:
#         setattr(
#             self,
#             key,
#         )
#     else:
#         setattr(self, key, self.datasets[name][key])
