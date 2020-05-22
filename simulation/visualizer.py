import altair as alt
import pandas as pd
import numpy as np

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.output import Output


class Visualizer(object):
    def __init__(self, dataset: Dataset, output: Output):
        self.dataset = dataset
        self.output_path = output.output_path
        self.df = output.df
        self.color = dict(
            zip("bprwg", ["#3498db", "#9b59b6", "#e74c3c", "#ffffff", "#ffffff"])
        )  ##95a5a6

    def color_array(self, i, color):
        if isinstance(color, list):
            return [color[0]] * i["amount"] if i["color"] else [color[1]] * i["amount"]
        else:
            return [color] * i

    def visualize(self):
        s = pd.Series(
            self.df["infection_date"].values - np.array(self.dataset.start_date)
        ).dt.days.apply(self.color_array, args=("g",))
        print(s)
        i = (
            (
                pd.to_datetime(self.df["expiration_date"])
                - pd.to_datetime(self.df["infection_date"])
            )
            .dt.days.to_frame(name="amount")
            .join(self.df["color"])
            .apply(self.color_array, args=(["p", "b"],), axis=1)
        )
        print(i)
        si = s.toframe().join(i)
        print(si)
        mylen = np.vectorize(len)
        r = (
            pd.Series(self.dataset.period - mylen(si))
            .to_frame(name="amount")
            .join(self.df["color"])
            .apply(self.color_array, args=(["r", "w"],), axis=1)
        )
        sir = si + r
        print(sir)
        summed = (
            sir.apply(pd.Series)
            .apply(pd.Series.value_counts)
            .reset_index()
            .melt(id_vars="index", value_vars=range(self.period))
        )
        summed["order"] = summed["index"].replace(
            {val: i for i, val in enumerate(color.keys())}
        )
        print(summed)
        chart = (
            alt.Chart(summed)
            .mark_bar()
            .encode(
                x="variable:O",
                y="value",
                color=alt.Color(
                    "index",
                    scale=alt.Scale(
                        domain=list(color.keys()), range=list(color.values())
                    ),
                ),
                order="order:O",
            )
        )
        chart.save(OUTPUT_FOLDER / f"{self.output_path}.png")
