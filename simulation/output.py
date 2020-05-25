from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.helpers import timing


class Output(object):
    def __init__(self, dataset: Dataset, output_filename="output"):
        self.reset()
        self.dataset = dataset
        self.output_filname = output_filename
        self.output_path = Path(OUTPUT_FOLDER / f"{output_filename}.csv")
        self.summed = []

    def reset(self):
        self.df = pd.DataFrame(
            columns=["age_group", "color", "infection_date", "expiration_date"]
        )
        self.df.index.name = "id"

    def display(self):
        print(self.df.to_string())

    def shape(self):
        print(self.df.shape)

    def export(self, name=None, summed=True):
        path = Path(OUTPUT_FOLDER / f"{name}.csv") if name else self.output_path
        if summed:
            self.average.to_csv(path, index=False)
        else:
            self.df.to_csv(path)

    def append(self, new_df):
        self.df = self.df.append(new_df, verify_integrity=True)
        self.df.index.name = "id"

    def _color_array(self, i, color):
        if isinstance(color, list):
            return [color[0]] * i[0] if i[1] else [color[1]] * i[0]
        else:
            return [color] * i

    def _color_lists(self, array: np.array, colors: Union[list, str]):
        return pd.concat(
            [pd.Series(array), self.df["color"].reset_index(drop=True)],
            axis=1,
            ignore_index=True,
        ).apply(self._color_array, args=(colors,), axis=1)

    @timing
    def sum_output(self):
        s = pd.Series(
            self.df["infection_date"].values - np.array(self.dataset.start_date)
        ).dt.days.apply(self._color_array, args=("g",))
        i = self._color_lists(
            (self.df["expiration_date"].values - self.df["infection_date"].values)
            .astype("timedelta64[D]")
            .astype(int),
            ["p", "b"],
        )
        r = self._color_lists(
            self.dataset.period - np.vectorize(len)(s + i), ["r", "w"]
        )
        return (
            (s + i + r)
            .apply(pd.Series)
            .apply(pd.Series.value_counts)
            .reset_index()
            .rename(columns={"index": "color"})
            .melt(
                id_vars="color",
                value_vars=range(self.dataset.period),
                value_name="amount",
                var_name=self.dataset.interval,
            )
        )

    def average_outputs(self):
        self.average = (
            pd.concat(self.summed)
            .groupby(["color", "day"])["amount"]
            .mean()
            .reset_index()
        )
