from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, Optional

import altair as alt  # type: ignore
import pandas as pd  # type: ignore

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.building_blocks import BasicBlock

alt.data_transformers.disable_max_rows()


class Visualizer(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, save: bool = False):
        super().__init__(dataset=dataset, task=task)
        self.save = save
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

    def _save_chart(self, chart: alt.Chart, suffix: str = None):
        if self.save:
            chart.save(
                str(
                    OUTPUT_FOLDER
                    / f"{self.task.id}{f'_{suffix}' if suffix else ''}.html"
                ),
                format="html",
            )

    def stacked_bar(
        self,
        df: pd.DataFrame,
        param: str = None,
        include_green: bool = True,
        interactive: bool = False,
    ) -> alt.Chart:
        """
        Parameters
        ----------
        df : pd.DataFrame
            {dataset.interval} | amount | color
        """
        df["order"] = df["color"].replace(
            {val: i for i, val in enumerate(self.colors.keys())}
        )
        if not include_green:
            df = df[df["color"] != "green"]
            self.colors.pop("green")
        df["color"] = df["color"].apply(self.states.decrypt_states)
        df["amount"] = df["amount"] / self.dataset.nodes
        # add to chart if title wanted: , **({} if got_input else {"title": self.dataset.name}))
        domain = [self.states.decrypt_states(c) for c in self.colors.keys()]
        chart = (
            alt.Chart(df)
            .mark_bar(size=(9 if self.dataset.period > 30 else 15))
            .encode(
                x=f"{self.dataset.interval}:O",
                y=alt.Y(
                    "amount:Q",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=(0, 1)),
                ),
                color=alt.Color(
                    "color",
                    scale=alt.Scale(domain=domain, range=list(self.colors.values())),
                ),
                order="order:O",
                tooltip=[
                    "color",
                    alt.Tooltip("amount:Q", format="%"),
                    f"{self.dataset.interval}",
                ],
            )
        )
        if not interactive and param:
            chart = (
                chart.facet(facet="step:N", columns=1)
                .properties(title=param)
                .configure_title(anchor="middle")
            )
        self.colors["green"] = "#09ab3b"
        self._save_chart(chart, suffix=param)
        return chart

    def boxplot(self, df: pd.DataFrame) -> alt.Chart:
        param = df.iloc[0]["parameter"].split("__")[0]
        range_ = self.task["sensitivity"]["ranges"][param]
        width = (range_["max"] - range_["min"]) * 20 / range_["step"]
        chart = (
            alt.Chart(df)
            .mark_boxplot()
            .encode(
                x=alt.X("step:N", title=None),
                y=alt.Y(
                    "value:Q",
                    title=None,
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=(0, 1)),
                ),
                color=alt.Color("metric", sort="ascending"),
            )
            .properties(height=300, width=width)
        )
        return chart

    def boxplots(self, df: pd.DataFrame) -> alt.HConcatChart:
        horizontal = []
        for param in df["parameter"].drop_duplicates():
            vertical = []
            for _, g in df[df["parameter"] == param].groupby("metric"):
                vertical.append(self.boxplot(g))
            horizontal.append(
                alt.vconcat(*vertical).properties(title=" ".join(param.split("__")))
            )
        chart = alt.hconcat(*horizontal)
        self._save_chart(chart, "sensitivity")
        return chart