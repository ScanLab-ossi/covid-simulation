from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING
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
from simulation.metrics import Metrics

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
        self.df = pd.DataFrame(
            [], columns=["infection_date", "days_left", "color"]  # , "age"]
        ).rename_axis(index="source")
        self.filename: str = str(task.id)
        self.summed = {}

    def __len__(self):
        if self.df == None:
            return 0
        else:
            return len(self.df.index)

    def sum_output(self):
        metrics = Metrics()
        df = pd.DataFrame(self.summed).T.rename_axis("day", axis="index").fillna(0)
        for metric in metrics.states_with_duration:
            if metric not in df.columns:
                df[metric] = 0
        return df

    def value_counts(self, day):
        self.summed[day] = dict(self.df["color"].value_counts())
        self.summed[day]["green"] = self.dataset.nodes - sum(self.summed[day].values())
        daily = self.df[self.df["infection_date"] == day]
        self.summed[day]["infected_daily"] = np.count_nonzero(daily)
        self.summed[day]["daily_infectors"] = (
            len(set.union(*daily["infectors"])) if "infectors" in daily.columns else 0
        )


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
        self.summed_list = [output.sum_output() for output in self.batch]
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
