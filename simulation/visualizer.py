import altair as alt  # type: ignore
from copy import copy
import pandas as pd  # type: ignore
from typing import Union, Optional

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.output import Batch, MultiBatch
from simulation.task import Task
from simulation.building_blocks import BasicBlock
from simulation.metrics import Metrics


class Visualizer(BasicBlock):
    def __init__(
        self,
        task: Task,
        dataset: Dataset,
        batches: Union[Batch, MultiBatch],
        save: bool = False,
    ):
        super().__init__(dataset=dataset, task=task)
        self.batches = batches
        self.save = save
        self.filename = str(self.task.id)
        self.colors = {
            "blue": "#0068c9",
            "purple_red": "#ff2b2b",
            "purple_pink": "#70349e",
            "pink": "#e83e8c",
            "stable_black": "#fd7e14",
            "stable_white": "#faca2b",
            "intensive_care_black": "#555867",
            "intensive_care_white": "#a3a8b4",
            "black": "#292a31",
            "white": "#e7e8e9",
            "green": "#09ab3b",
        }
        # old colors: "bprkwg", ["#17a2b8", "#6f42c1", "#ff2b2b", "#262730", "#f0f2f6", "#09ab3b"],
        # old old colors: ["#3498db", "#9b59b6", "#e74c3c", "#000000", "#dddddd", "#4daf4a"],

    def _save_chart(self, chart: alt.Chart, suffix: str = None):
        if self.save:
            chart.save(
                str(
                    OUTPUT_FOLDER
                    / f"{self.filename}{f'_{suffix}' if suffix else ''}.html"
                ),
                format="html",
            )

    def visualize(
        self, df: Optional[pd.DataFrame] = None, include_green: bool = True
    ) -> alt.Chart:
        got_input = isinstance(df, pd.DataFrame)
        if got_input:
            summed = df.drop(columns=["infected_daily", "daily_infectors"]).melt(
                id_vars="day", var_name="color", value_name="amount"
            )
        else:
            self.batches.average_outputs()
            summed = (
                self.batches.average.drop(columns=["infected_daily", "daily_infectors"])
                .reset_index()
                .melt(id_vars="day", var_name="color", value_name="amount")
            )
        summed["order"] = summed["color"].replace(
            {val: i for i, val in enumerate(self.colors.keys())}
        )
        if not include_green:
            summed = summed[summed["color"] != "green"]
            self.colors.pop("green")
        summed["color"] = summed["color"].apply(Metrics.decrypt_colors)
        # add to chart if title wanted: , **({} if got_input else {"title": self.dataset.name}))
        chart = (
            alt.Chart(summed)
            .mark_bar(size=(9 if self.dataset.period > 30 else 15))
            .encode(
                x=f"{self.dataset.interval}:O",
                y="amount",
                color=alt.Color(
                    "color",
                    scale=alt.Scale(
                        domain=[Metrics.decrypt_colors(c) for c in self.colors.keys()],
                        range=list(self.colors.values()),
                    ),
                ),
                order="order:O",
                tooltip=["color", "amount", f"{self.dataset.interval}"],
            )
        )
        self._save_chart(chart)
        self.colors["green"] = "#09ab3b"
        return chart

    # def variance_boxplot(self) -> alt.FacetChart:
    #     chart = (
    #         alt.Chart(self.batches.concated)
    #         .mark_boxplot()
    #         .encode(
    #             x="day:O",
    #             y="amount:Q",
    #             color=alt.Color(
    #                 "color",
    #                 scale=alt.Scale(
    #                     domain=list(self.colors.keys()),
    #                     range=list(self.colors.values()),
    #                 ),
    #             ),
    #         )
    #         .properties(width=800, height=400)
    #         .facet(row="color")
    #     )
    #     if self.save:
    #         chart.save(
    #             str(OUTPUT_FOLDER / f"{self.filename}_variance.html"), format="html"
    #         )
    #     return chart

    def _get_base(self, parameter):
        if parameter not in self.task:
            sub = [k for k in self.task["paths"][parameter].keys() if k[0] == "d"][0]
            base = self.task["paths"][parameter][sub][0]
        else:
            base = self.task[parameter]
        return base

    def sensitivity_boxplots(
        self,
        df: Optional[pd.DataFrame] = None,
        steps: bool = True,
    ) -> alt.FacetChart:  # metric: str = None
        got_input = isinstance(df, pd.DataFrame)
        df = df if got_input else self.batches.summed
        print(df)
        df["base"] = df["parameter"].apply(self._get_base)
        if not steps:
            df["step"] = (
                df["step"].apply(eval)
                * df["parameter"].apply(
                    lambda x: self.task["sensitivity"]["ranges"][x]["step"]
                )
                + df["base"]
            )
        df["display_parameter"] = df[["parameter", "base"]].apply(
            lambda x: "{}: {}".format(x[0], x[1]), axis=1
        )
        sort = {"sort": sorted(set(df["step"].tolist()), key=eval)} if steps else {}
        chart = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                x=alt.X("step:N", title=None, **sort),
                y=alt.Y("value:Q", title=None),
                color=alt.Color("metric", sort="ascending")
                # color=alt.Color(
                #     "parameter",
                #     scale=alt.Scale(
                #         domain=list(self.colors.keys()),
                #         range=list(self.colors.values()),
                #     ),
                # ),
            )
            .properties(height=300, width=150)
            .facet(column="display_parameter", row="metric")
        )
        self._save_chart(chart, "sensitivity")
        return chart
