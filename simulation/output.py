from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union
from datetime import datetime

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.helpers import timing


class Output(object):
    def __init__(self, dataset: Dataset, task: Task):
        self.reset()
        self.dataset = dataset
        self.batch = []
        self.filename = str(task.id)
        self.colors = list("bprkwg")

    def reset(self):
        self.df = pd.DataFrame(
            columns=[
                "age_group",
                "color",
                "infection_date",
                "transition_date",
                "expiration_date",
                "final_state",
            ]
        )
        self.df.index.name = "id"

    def export(
        self,
        filename: Union[str, None] = "output",
        how: str = "average",
        pickle: bool = False,
    ):
        # average, concated, df
        if not hasattr(self, how):
            if how == "concated":
                self.concat_outputs()
            elif how == "averaged":
                self.average_outputs()
            else:
                raise AttributeError(f'you haven\'t created attribute "{how}" yet')
        filename = self.filename + (f"_{how}" if settings["LOCAL"] else "")
        self.csv_path = Path(OUTPUT_FOLDER / f"{filename}.csv")
        getattr(self, how).to_csv(self.csv_path, index=(False if how != "df" else True))
        if pickle:
            self.pickle_path = Path(OUTPUT_FOLDER / f"{filename}.pkl")
            getattr(self, how).to_pickle(self.pickle_path)

    def append(self, new_df):
        self.df = self.df.append(new_df, verify_integrity=True)
        self.df.index.name = "id"

    def _add_missing(self, df):
        for k in self.colors:
            if k not in df.index:
                df = df.append(
                    pd.DataFrame.from_dict(
                        {k: [0] * self.dataset.period}, orient="index"
                    )
                )
        return df

    def _color_array(self, i, color):
        if isinstance(color, list):
            return [color[0]] * i[0] if i[1] else [color[1]] * i[0]
        else:
            return [color] * i

    def _color_lists(self, a: np.array, colors: pd.Series, letters: Union[list, str]):
        return pd.concat([pd.Series(a), colors], axis=1, ignore_index=True).apply(
            self._color_array, args=(letters,), axis=1
        )

    @timing
    def sum_output(self, df):
        # s2i = start_to_infection, i2t = infection_to_transition,
        # t2e = transition_to_expiration, e2ft = expiration_to_final_state
        # u = uninfected
        colors = df["color"].reset_index(drop=True)
        s2i = pd.Series(
            df["infection_date"].values - np.array(self.dataset.start_date)
        ).dt.days.apply(self._color_array, args=("g",))
        i2t = self._color_lists(
            (df["transition_date"].values - df["infection_date"].values)
            .astype("timedelta64[D]")
            .astype(int),
            colors,
            ["p", "b"],
        )
        t2e = self._color_lists(
            (df["expiration_date"].values - df["transition_date"].values)
            .astype("timedelta64[D]")
            .astype(int),
            colors,
            ["r", "w"],
        )
        e2ft = pd.Series(
            df["final_state"].apply(list).values
            * (self.dataset.period - np.vectorize(len)(s2i + i2t + t2e))
        )
        u = pd.Series(
            [["g"] * self.dataset.period for _ in range(self.dataset.nodes - len(df))]
        )
        return (
            (s2i + i2t + t2e + e2ft)
            .append(u)
            .apply(pd.Series)
            .apply(pd.Series.value_counts)
            .fillna(0)
            .pipe(self._add_missing)
            .reset_index()
            .rename(columns={"index": "color"})
            .melt(
                id_vars="color",
                value_vars=range(self.dataset.period),
                value_name="amount",
                var_name=self.dataset.interval,
            )
        )

    def sum_and_concat_outputs(self):
        self.concated = pd.concat([self.sum_output(df) for df in self.batch])

    def average_outputs(self):
        if not hasattr(self, "concated"):
            self.sum_and_concat_outputs()
        self.average = (
            self.concated.groupby(["color", "day"])["amount"].mean().reset_index()
        )
