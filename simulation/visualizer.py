from __future__ import annotations
from typing import TYPE_CHECKING
from typing import Union, Optional

import altair as alt
from altair.vegalite.v4.schema.channels import Tooltip  # type: ignore
import pandas as pd
from numpy import random

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.building_blocks import BasicBlock

alt.data_transformers.disable_max_rows()


class Visualizer(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, save: bool = False):
        super().__init__(dataset=dataset, task=task)
        self.save = save
        self.old_colors = {
            "blue": "#0068c9",
            "purple": "#70349e",
            "purple_red": "#ff2b2b",
            "purple_pink": "#70349e",
            "pink": "#e83e8c",
            "red": "#ff2b2b",
            "stable_black": "#fd7e14",
            "stable_white": "#faca2b",
            "intensive_care_black": "#555867",
            "intensive_care_white": "#a3a8b4",
            "black": "#292a31",
            "white": "#e7e8e9",
            "green": "#09ab3b",
        }
        self.colors = {
            "blue": "#2166ac",
            "purple_red": "#d6604d",
            "purple_pink": "#d6604d",
            "purple": "#d6604d",  #
            "pink": "#d6604d",
            "red": "#b2182b",  #
            "stable_black": "#b2182b",
            "stable_white": "#b2182b",
            "intensive_care_black": "#b2182b",
            "intensive_care_white": "#b2182b",
            "black": "#515151",
            "white": "#e7e8e9",
            "green": "#92c5de",
        }
        self.facet_header = alt.Header(labelFontWeight="bold", labelFontSize=12)

    def _save_chart(self, chart: alt.Chart, suffix: str = None):
        if self.save:
            chart.save(
                str(
                    OUTPUT_FOLDER
                    / f"{self.task.id}{f'_{suffix}' if suffix else ''}.html"
                ),
                format="html",
            )

    def _prep_for_stacked(
        self,
        df: pd.DataFrame,
        include_green: bool,
        simplified: bool,
    ) -> pd.DataFrame:
        if not include_green:
            df = df[df["color"] != "green"]
            self.colors.pop("green")
        if simplified:
            df["color"] = (
                df["color"]
                .replace(self.states.get_filter("red")["regex"], "red", regex=True)
                .replace("purple.*", "purple", regex=True)
            )
            self.colors = {
                k: v
                for k, v in self.colors.items()
                if k in list(df["color"].drop_duplicates())
            }
        df["order"] = df["color"].replace(
            {val: i for i, val in enumerate(self.colors.keys())}
        )
        df["color"] = df["color"].replace(self.states.color_to_state)
        df["amount"] = df["amount"] / self.dataset.nodes
        return df

    def stacked_bar(
        self,
        df: pd.DataFrame,
        param: str = None,
        include_green: bool = True,
        interactive: bool = False,
        simplified: bool = True,
    ) -> alt.Chart:
        """
        Parameters
        ----------
        df : pd.DataFrame
            day | amount | color
        """
        df = self._prep_for_stacked(df, include_green, simplified)
        domain = [self.states.color_to_state[c] for c in self.colors]
        chart = (
            alt.Chart(df)
            .mark_bar(size=15)
            .encode(
                x=f"day:O",
                y=alt.Y(
                    "amount:Q",
                    axis=alt.Axis(format="%", grid=False),
                    scale=alt.Scale(domain=(0, 1)),
                ),
                color=alt.Color(
                    "color",
                    scale=alt.Scale(domain=domain, range=list(self.colors.values())),
                ),
                order="order:O",
                tooltip=["color", alt.Tooltip("amount:Q", format="%"), "day"],
            )
        )
        if not interactive and param:
            chart = (
                chart.facet(facet="step:N", columns=1)
                .properties(title=param)
                .configure_title(anchor="middle")
            )
        if not include_green:
            self.colors["green"] = "#09ab3b"
        self._save_chart(chart, suffix=param)
        return chart

    def _prep_for_line(self, df: pd.DataFrame) -> pd.DataFrame:
        if "variant" not in df.columns:
            df["variant"] = "variant_a"
        df = df[df["color"].isin(["infected", "infected_daily", "sick"])].reset_index(
            drop=True
        )
        df["variant"] = df["variant"].replace(self.variants.js)
        df["amount"] = df["amount"] / self.dataset.nodes
        return df

    def line(self, df: pd.DataFrame, param: Optional[str] = None) -> alt.Chart:
        """
        Parameters
        ----------
        df : pd.DataFrame
            day | variant | color | amount
        """
        df = self._prep_for_line(df)
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                alt.X("day:N"),
                alt.Y("amount:Q", axis=alt.Axis(format="%"), title=None),
                color=alt.Color("variant:N"),
                tooltip=["variant", alt.Tooltip("amount:Q", format="%"), "day"],
            )
            .facet(
                column=alt.Column("step:O", title=None, header=self.facet_header),
                row=alt.Row("color:O", title=None, header=self.facet_header),
            )
            .resolve_scale(x="independent")
        )
        self._save_chart(chart, "line")
        return chart

    def boxplot(self, df: pd.DataFrame) -> alt.Chart:
        try:
            color = [v for k, v in self.colors.items() if k in df["metric"].iloc[0]][0]
        except IndexError:
            color = random.choice(list(self.colors.values()), 1)
        chart = (
            alt.LayerChart(df)
            .encode(x="step:O")
            .add_layers(
                alt.Chart()
                .mark_boxplot(median=False)  # , color=color)
                .encode(
                    x=alt.X("step:N", title=None),
                    y=alt.Y(
                        "value:Q",
                        title=df["metric"].iloc[0],
                        axis=alt.Axis(format="%"),
                        scale=alt.Scale(domain=(0, 1), clamp=True),
                    ),
                ),
                alt.Chart()
                .transform_aggregate(
                    mean="mean(value)",
                    groupby=["step"],
                )
                .mark_tick(color="white", width=15, thickness=2.5)
                .encode(y="mean:Q", tooltip=alt.Tooltip("mean:Q", format="%")),
            )
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
