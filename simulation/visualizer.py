import altair as alt
import pandas as pd

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.building_blocks import BasicBlock

alt.data_transformers.disable_max_rows()


class Visualizer(BasicBlock):
    """
    we change values of `param` and check outcome on `metric`"
    """

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
            "purple": "#d6604d",
            "pink": "#d6604d",
            "red": "#b2182b",
            "stable_black": "#b2182b",
            "stable_white": "#b2182b",
            "intensive_care_black": "#b2182b",
            "intensive_care_white": "#b2182b",
            "black": "#515151",
            "white": "#e7e8e9",
            "green": "#92c5de",
        }

    def _save_chart(self, chart: alt.Chart, suffix: str = None):
        if self.save:
            chart.save(
                OUTPUT_FOLDER / f"{self.task.id}{f'_{suffix}' if suffix else ''}.html",
                format="html",
            )

    def _prep_for_stacked(
        self,
        df: pd.DataFrame,
        include_green: bool,
        simplified: bool,
    ) -> pd.DataFrame:
        if not include_green:
            df = df[df["state"] != "green"]
            self.colors.pop("green")
        if simplified:
            df["state"] = (
                df["state"]
                .replace(self.states.get_filter("red")["regex"], "red", regex=True)
                .replace("purple.*", "purple", regex=True)
            )
            self.colors = {
                k: v
                for k, v in self.colors.items()
                if k in list(df["state"].drop_duplicates())
            }
        df["order"] = df["state"].replace(
            {val: i for i, val in enumerate(self.colors.keys())}
        )
        df["state"] = df["state"].replace(self.states.color_to_state)
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
            day | amount | state
        """
        df = self._prep_for_stacked(df, include_green, simplified)
        domain = [self.states.color_to_state[c] for c in self.colors]
        chart = (
            alt.Chart(df)
            .mark_bar(size=15)
            .encode(
                x=alt.X(f"day:O", title="Time Window"),
                y=alt.Y(
                    "amount:Q",
                    title="Amount",
                    axis=alt.Axis(format="%", grid=False),
                    scale=alt.Scale(domain=(0, 1)),
                ),
                color=alt.Color(
                    "state",
                    title="State",
                    scale=alt.Scale(domain=domain, range=list(self.colors.values())),
                ),
                order="order:O",
                tooltip=["state", alt.Tooltip("amount:Q", format="%"), "day"],
            )
            .configure_axis(labelFontSize=14, titleFontSize=16)
            .configure_legend(labelFontSize=14, titleFontSize=16)
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

    def point_with_ci(self, df, save=True):
        df = self._prep_for_stacked(df, include_green=True, simplified=True)
        # df = df[df["state"] == "sick"]  # FIXME: remove
        domain = [self.states.color_to_state[c] for c in self.colors]
        x = alt.X("day:O", title="Time Window")
        y = alt.Y("amount:Q", title="Amount")
        color = alt.Color(
            "state",
            title="State",
            scale=alt.Scale(domain=domain, range=list(self.colors.values())),
        )
        # color = alt.Color("step:N")  # FIXME: remove
        error_bars = (
            alt.Chart(df).mark_errorbar(extent="ci").encode(x=x, y=y, color=color)
        )
        # FIXME: remove step
        points = (
            alt.Chart(df.groupby(["day", "state"], as_index=False).mean())
            .mark_point()
            .encode(x=x, y=y, color=color)
        )
        chart = (
            (error_bars + points)
            .configure_axis(labelFontSize=14, titleFontSize=16)
            .configure_legend(labelFontSize=14, titleFontSize=16)
        )

        if save:
            self._save_chart(chart, f"point_with_ci")
        return chart

    def _prep_for_line(self, df: pd.DataFrame) -> pd.DataFrame:
        if "variant" not in df.columns:
            df["variant"] = "variant_a"
        df = df[df["state"].isin(["infected", "daily_infected", "sick"])].reset_index(
            drop=True
        )
        # df["variant"] = df["variant"].replace(self.variants.params)
        df["amount"] = df["amount"] / self.dataset.nodes
        return df

    def line(
        self, df: pd.DataFrame, metric: str | None = None, save: bool = False
    ) -> alt.Chart:
        """
        Parameters
        ----------
        df : pd.DataFrame
            day | variant | state | amount
        """
        df = self._prep_for_line(df)
        if metric:
            df = df[df["state"] == metric]
        line = (
            alt.Chart()
            .transform_aggregate(amount="mean(amount)", groupby=["day", "variant"])
            .mark_line()
            .encode(
                x=alt.X("day", title="τ"),
                y=alt.Y("amount:Q", axis=alt.Axis(format="%"), title="Total infection"),
                color=alt.Color("variant:N", title="Variant"),
                tooltip=["variant", alt.Tooltip("amount:Q", format="%"), "day"],
            )
            .properties(width=150, height=150)
            # .resolve_scale(x="independent")
        )
        band = (
            alt.Chart()
            .mark_errorband(extent="ci")
            .encode(
                x=alt.X("day", title="τ"),
                y=alt.Y("amount", title="Total infection"),
                color="variant",
            )
        )
        chart = alt.layer(line, band, data=df)
        # chart = (
        #     alt.Chart(df)
        #     .mark_line()
        #     .encode(
        #         alt.X("day", title="τ"),
        #         alt.Y("amount:Q", axis=alt.Axis(format="%"), title="Total infection"),
        #         color=alt.Color("variant:N", title="Variant"),
        #         tooltip=["variant", alt.Tooltip("amount:Q", format="%"), "day"],
        #     )
        #     .properties(width=150, height=150)
        #     # .resolve_scale(x="independent")
        # )
        if save:
            self._save_chart(chart, f"line_{metric}")
        return chart

    def facet_line(self, df: pd.DataFrame, metric: str, param: str) -> alt.Chart:
        header = alt.Header(labelFontWeight="bold", labelFontSize=12)
        chart = self.line(df, metric=metric).facet(
            column=alt.Column("step_0:O", title=None, header=header, sort="descending"),
            row=alt.Row("step_1:O", title=None, header=header, sort="descending"),
            title=f"Changing {param}, showing progression of {metric}",
        )
        self._save_chart(chart, f"line_{metric}")
        return chart

    def boxplot(self, df: pd.DataFrame) -> alt.Chart:
        # try:
        #     color = [v for k, v in self.colors.items() if k in df["metric"].iloc[0]][0]
        # except IndexError:
        #     color = random.choice(list(self.colors.values()), 1)
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

    def heatmap(self, df: pd.DataFrame, metric: str, param: str) -> alt.Chart:
        chart = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=alt.X(
                    "step_0:O",
                    axis=alt.Axis(
                        orient="top", title=f"change in {param} for {self.variants[0]}"
                    ),
                ),
                y=alt.Y(
                    "step_1:O",
                    axis=alt.Axis(title=f"change in {param} for {self.variants[1]}"),
                ),
                color=alt.Color(
                    "amount:Q",
                    title=f"{metric} ratio (normalized)",
                ),
                tooltip=[alt.Tooltip("amount", title=f"ratio")],
            )
        )
        self._save_chart(chart, f"heatmap_{metric}")
        return chart
