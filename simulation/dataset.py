from datetime import datetime
import pandas as pd
import json

from simulation.constants import *
from simulation.google_cloud import GoogleCloud


class Dataset(object):
    def __init__(self, name):
        self.name = name
        with open(DATA_FOLDER / "datasets.json", "r") as f:
            datasets = json.load(f)
        for key in datasets[name]:
            if "date" in key:
                setattr(
                    self, key, datetime.strptime(datasets[name][key], "%Y-%m-%d").date()
                )
            else:
                setattr(self, key, datasets[name][key])
        self.period = (self.end_date - self.start_date).days

    def load_dataset(self, gcloud: GoogleCloud = None):
        if (self.storage == "csv") and not (DATA_FOLDER / f"{self.name}.csv").exists():
            gcloud.download(f"{self.name}.csv")
        self.data = pd.read_csv(
            DATA_FOLDER / f"{self.name}.csv", parse_dates=["datetime"]
        )