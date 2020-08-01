import altair as alt  # type: ignore
from copy import copy
import pandas as pd  # type: ignore
from typing import Union, Optional

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.output import Output
from simulation.task import Task
from simulation.building_blocks import OutputBasicBlock


class Visualizer(OutputBasicBlock):
    def __init__(self, output: Output, task: Task, dataset: Dataset):
        super().__init__(dataset=dataset, output=output, task=task)
        self.filename = str(self.task.id)
        self.colors = dict(
            zip(
                "bprkwg",
                ["#17a2b8", "#6f42c1", "#ff2b2b", "#262730", "#f0f2f6", "#09ab3b"],
            )
        )
        # old colors: ["#3498db", "#9b59b6", "#e74c3c", "#000000", "#dddddd", "#4daf4a"],

    def visualize(self, df: Optional[pd.DataFrame] = None) -> alt.Chart:
        got_input = isinstance(df, pd.DataFrame)
        summed = df if got_input else self.output.average
        summed["order"] = summed["color"].replace(
            {val: i for i, val in enumerate(self.colors.keys())}
        )
        chart = (
            alt.Chart(summed, **({} if got_input else {"title": self.dataset.name}))
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
        if not got_input:
            chart.save(str(OUTPUT_FOLDER / f"{self.filename}.html"), format="html")
        return chart

    def variance_boxplot(self) -> alt.FacetChart:
        chart = (
            alt.Chart(self.output.concated)
            .mark_boxplot()
            .encode(
                x="day:O",
                y="amount:Q",
                color=alt.Color(
                    "color",
                    scale=alt.Scale(
                        domain=list(self.colors.keys()),
                        range=list(self.colors.values()),
                    ),
                ),
            )
            .properties(width=800, height=400)
            .facet(row="color")
        )
        chart.save(str(OUTPUT_FOLDER / f"{self.filename}_variance.html"), format="html")
        return chart

    def sensitivity_boxplot(
        self, df: Union[pd.DataFrame, None] = None, grouping: str = "step"
    ) -> alt.FacetChart:
        got_input = isinstance(df, pd.DataFrame)
        metric = self.task["sensitivity"]["metric"]
        df = df if got_input else pd.concat(self.output.results)
        base_values = [
            f"{param}: {self.task[param]}"
            for param in sorted(self.task["sensitivity"]["params"])
        ]
        chart = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                y=f"{metric}:Q",
                color=alt.Color(
                    "parameter",
                    sort="ascending",
                    legend=alt.Legend(values=base_values),
                ),
            )
        )
        step_sort = sorted(set(df["step"].tolist()), key=eval)
        if grouping == "parameter":
            chart = chart.encode(x=alt.X("parameter", title=None)).facet(
                alt.Facet("step:O", sort=step_sort, spacing=5)
            )
        elif grouping == "step":
            chart = chart.encode(
                x=alt.X("step", title=None, sort=step_sort),
                y=f"{metric}:Q",
                color="parameter",
            ).facet("parameter:N")
        if not got_input:
            chart.save(
                str(OUTPUT_FOLDER / f"{self.filename}_sensitivity.html"), format="html"
            )
        return chart
