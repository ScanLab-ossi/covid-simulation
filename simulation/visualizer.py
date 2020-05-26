import altair as alt
from altair.expr import datum
from copy import copy

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.output import Output


class Visualizer(object):
    def __init__(self, output: Output):
        self.dataset = output.dataset
        self.output = output
        self.output_filename = output.output_filname
        self.colors = dict(
            zip("bprwg", ["#3498db", "#9b59b6", "#e74c3c", "#ffffff", "#ffffff"])
        )
        self.real_colors = (lambda a, b: a.update(b) or a)(
            self.colors, {"w": "#dddddd", "g": "#4daf4a"}
        )

    def visualize(self):
        summed = self.output.average
        summed["order"] = summed["color"].replace(
            {val: i for i, val in enumerate(self.colors.keys())}
        )
        chart = (
            alt.Chart(summed, title=self.dataset.name)
            .mark_bar()
            .encode(
                x=f"{self.dataset.interval}:O",
                y="amount",
                color=alt.Color(
                    "color",
                    scale=alt.Scale(
                        domain=list(self.colors.keys()),
                        range=list(self.colors.values()),
                    ),
                ),
                order="order:O",
                tooltip=["color", "amount", f"{self.dataset.interval}"],
            )
        )
        chart.save(str(OUTPUT_FOLDER / f"{self.output_filename}.html"), format="html")
        return chart

    def boxplot_variance(self):
        chart = (
            alt.Chart(self.output.concated)
            .mark_boxplot()
            .encode(
                x="day:O",
                y="amount:Q",
                color=alt.Color(
                    "color",
                    scale=alt.Scale(
                        domain=list(self.real_colors.keys()),
                        range=list(self.real_colors.values()),
                    ),
                ),
            )
            .properties(width=800, height=400)
            .facet(row="color")
        )
        chart.save(
            str(OUTPUT_FOLDER / f"{self.output_filename}_variance.html"), format="html"
        )
        return chart
