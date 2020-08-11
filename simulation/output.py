from __future__ import annotations
from abc import ABC
import bz2
import _pickle as cPickle
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict
from datetime import datetime

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.helpers import timing
from simulation.building_blocks import BasicBlock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.analysis import Analysis


class OutputBase(ABC):
    def __init__(self, task: Task):
        self.task = task
        self.filename: str = str(task.id)

    def export(
        self,
        filename: Optional[str] = None,
        what: str = "summed",
        format_: str = "csv",
    ):
        filename = self.filename if filename == None else filename
        # average, concated, df
        if not hasattr(self, what):
            raise AttributeError(f'you haven\'t created attribute "{what}" yet')
        self.export_path = (
            OUTPUT_FOLDER / f"{filename}.{'pbz2' if format_ == 'pickle' else 'csv'}"
        )
        if format_ == "pickle":
            with bz2.BZ2File(self.export_path, "w") as f:
                cPickle.dump(getattr(self, what), f)
        elif format_ == "csv":
            getattr(self, what).to_csv(
                self.export_path, index=(False if hasattr(self, "batches") else True)
            )

    def load_pickle(self, file_path: Path) -> Union[Batch, MultiBatch]:
        data = cPickle.load(bz2.BZ2File(file_path, "rb"))
        return data


class Output(BasicBlock):
    """
    result of one iteration
    """

    def __init__(self, dataset: Dataset, task: Task):
        super().__init__(dataset=dataset, task=task)
        self.reset()
        self.filename: str = str(task.id)
        self.colors: List[str] = list("bprkwg")

    def __len__(self):
        return len(self.df.index)

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

    def append(self, new_df: pd.DataFrame):
        self.df = self.df.append(new_df, verify_integrity=True)
        self.df.index.name = "id"

    def _add_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for k in self.colors:
            if k not in df.index:
                df = df.append(
                    pd.DataFrame.from_dict(
                        {k: [0] * self.dataset.period}, orient="index"
                    )
                )
        return df

    def _color_array(
        self, i: Union[List[int], int], color: Union[List[str], str]
    ) -> List[str]:
        if isinstance(color, list):
            return [color[0]] * i[0] if i[1] else [color[1]] * i[0]
        else:
            return [color] * i

    def _color_lists(
        self, a: np.array, colors: pd.Series, letters: Union[list, str]
    ) -> pd.DataFrame:
        return pd.concat([pd.Series(a), colors], axis=1, ignore_index=True).apply(
            self._color_array, args=(letters,), axis=1
        )

    @timing
    def sum_output(self, df: pd.DataFrame, pivot=False) -> pd.DataFrame:
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
        res = (
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
        return res.pivot(index="day", columns="color")["amount"] if pivot else res

    def infections(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = pd.date_range(self.dataset.start_date, self.dataset.end_date)
        return (
            df.groupby("infection_date")
            .agg(
                infectors=pd.NamedAgg(
                    column="infector", aggfunc=lambda x: len(set().union(*x.dropna())),
                ),
                infected=pd.NamedAgg(column="final_state", aggfunc="count"),
            )
            .reindex(idx, fill_value=0)
            .reset_index(drop=True)
        )

    def sick(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df["b"] + df["p"] + df["r"]).to_frame("sick")


class Batch(OutputBase):
    def __init__(self, task: Task):
        super().__init__(task=task)
        self.batch: List[Optional[Output]] = []

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.batch):
            self.n += 1
            return self.batch[self.n - 1]
        else:
            raise StopIteration

    def append_df(self, output: Output):
        self.batch.append(output)

    def average_outputs(self):
        self.average = self.summed.groupby("day").mean()

    def sum_all_and_concat(self):
        self.summed_list = []
        for output in self.batch:
            colors = output.sum_output(output.df, pivot=True)
            sick = output.sick(colors)
            infections = output.infections(output.df)
            self.summed_list.append(colors.join(sick).join(infections))
        self.summed = pd.concat(self.summed_list)

    def load(self, file_path=None, format_="csv"):
        if not file_path:
            file_path = OUTPUT_FOLDER / f"{self.task.id}.{format_}"
        if format_ == "csv":
            if not file_path:
                file_path = OUTPUT_FOLDER / f"{self.task.id}.csv"
            self.summed = pd.read_csv(file_path).set_index("day")
            self.summed_list = np.array_split(self.summed, self.task["ITERATIONS"])
        elif format_ == "pickle":
            return self.load_pickle(file_path)


class MultiBatch(OutputBase):
    """
    {(param, value, relative_steps): batch}
    """

    def __init__(self, task: Task):
        super().__init__(task=task)
        self.batches: Dict[Tuple[str, Union[int, float], str], Batch] = {}

    def append_batch(
        self, batch: Batch, param: str, value: Union[int, float], relative_steps: str
    ):
        self.batches[(param, value, relative_steps)] = batch

    def analysis_sum(self, analysis: Analysis) -> pd.DataFrame:
        results = []
        for k, batch in self.batches.items():
            batch.sum_all_and_concat()
            for metric in self.task["sensitivity"]["metrics"]:
                param, value, relative_steps = k
                results.append(
                    analysis.count(batch, avg=False, **metric).assign(
                        **{"step": relative_steps, "parameter": param,}
                    )
                )
        self.summed = pd.concat(results)

    def load(self, file_path=None, format_="csv"):
        if not file_path:
            file_path = OUTPUT_FOLDER / f"{self.task.id}.{format_}"
        if format_ == "csv":
            self.summed = pd.read_csv(file_path, dtype={"step": str})
        elif format_ == "pickle":
            return self.load_pickle(file_path)
