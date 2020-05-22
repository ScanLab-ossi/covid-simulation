from pathlib import Path
import pandas as pd

from simulation.constants import *


class Output(object):
    def __init__(self, output_filename="output"):
        self.df = pd.DataFrame(
            columns=["age_group", "color", "infection_date", "expiration_date"]
        )
        self.df.index.name = "id"
        self.output_path = Path(OUTPUT_FOLDER / f"{output_filename}.csv")

    def display(self):
        print(self.df.to_string())

    def shape(self):
        print(self.df.shape)

    def export(self):
        self.df.to_csv(self.output_path)

    def append(self, new_df):
        self.df = self.df.append(new_df, verify_integrity=True)
        self.df.index.name = "id"
