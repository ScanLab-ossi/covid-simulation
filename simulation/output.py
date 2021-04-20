from __future__ import annotations
from abc import ABC
import gzip, json, pickle
from pathlib import Path
from typing import Union, List, Optional, Dict

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.task import Task
from simulation.helpers import timing
from simulation.building_blocks import BasicBlock
from simulation.states import States
from simulation.visualizer import Visualizer
from simulation.analysis import Analysis


class OutputBase(ABC):
    def __init__(self, dataset: Dataset, task: Task):
        self.dataset = dataset
        self.task = task
        self.visualizer = Visualizer(task, dataset, save=True)

    def export(self, *what: str, table_format: str = "csv"):
        for attr in what:
            d = getattr(self, attr)
            if not hasattr(self, attr):
                raise AttributeError(f'you haven\'t created attribute "{attr}" yet')
            if isinstance(d, dict):
                if isinstance(self, MultiBatch):
                    for param, s in d.items():
                        for step, output in s.items():
                            d[param][step] = output.mean_and_std
                with open(OUTPUT_FOLDER / f"{self.task.id}.json", "w") as fp:
                    json.dump(d, fp)
            if isinstance(d, pd.DataFrame):
                if table_format == "pickle":
                    with gzip.GzipFile(OUTPUT_FOLDER / f"{self.task.id}.pgz", "w") as f:
                        pickle.dump(d, f)
                elif table_format == "csv":
                    d.to_csv(
                        OUTPUT_FOLDER / f"{self.task.id}.csv",
                        index=(False if hasattr(self, "batches") else True),
                    )

    def load_pickle(self, file_path: Path) -> Union[Batch, MultiBatch]:
        data = pickle.load(gzip.GzipFile(file_path, "rb"))
        return data


class Output(BasicBlock):
    """
    Result of one iteration

    Attributes
    ----------
    df : pd.DataFrame
        infection_date | days_left | color | variant | infector
    summed : dict
    variant_summed :

    """

    def __init__(self, dataset: Dataset, task: Task, loaded: Optional[dict] = None):
        super().__init__(dataset=dataset, task=task)
        self.df = pd.DataFrame(
            [],
            columns=[
                "infection_date",
                "days_left",
                "color",
                "variant",
                "infector",
            ],
        ).rename_axis(index="source")
        self.summed = loaded if loaded else {}
        self.variant_summed = {}
        self.states = States()  # TODO: move states to BasicBlock

    def __len__(self):
        return len(self.df.index)

    def sum_output(self) -> pd.DataFrame:
        df = pd.DataFrame(self.summed).T.rename_axis("day", axis="index").fillna(0)
        for metric in self.states.states_with_duration:
            if metric not in df.columns:
                df[metric] = 0
        return df

    def value_counts(self, day: int):
        self.summed[day] = dict(self.df["color"].value_counts())
        self.summed[day]["green"] = self.dataset.nodes - sum(self.summed[day].values())
        daily = self.df[self.df["infection_date"] == day]
        self.summed[day]["infected_daily"] = len(daily)
        notna_infectors = daily[daily["infector"].notna()]["infector"]
        self.summed[day]["daily_infectors"] = (
            len(set.union(*notna_infectors)) if len(notna_infectors) > 0 else 0
        )
        self.variant_summed[day] = {}
        self.variant_summed[day]["infected"] = (
            self.df.groupby("variant")["color"].count().to_dict()
        )
        self.variant_summed[day]["infected_daily"] = (
            daily.groupby("variant")["color"].count().to_dict()
        )
        self.variant_summed[day]["sick"] = (
            self.df[self.df["color"].isin(self.states.sick_states)]
            .groupby("variant")["color"]
            .count()
            .to_dict()
        )
        pass


class Batch(OutputBase):
    """
    Results of all iterations with the same config


    Attributes
    ----------
    batch : List[Output]
        list of one Output per iteration
    mean_and_std : Dict[str, Dict]
        {mean: {}, std: {}}
        where each is a dict of mean / std value per state per day of all iterations
    summed_output_list : List[pd.DataFrame]

    Methods
    -------
    append_output()
    export()
    load()
    sum_batch() :
        add attrs mean_and_std, summed_output_list

    Inherited
    ---------
    Attributes: task, dataset
    Methods: export()
    """

    def __init__(self, dataset: Dataset, task: Task, init: Optional[dict] = None):
        super().__init__(dataset=dataset, task=task)
        self.mean_and_std = init if init else {}
        self.batch = []

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.batch):
            self.n += 1
            return self.batch[self.n - 1]
        else:
            raise StopIteration

    def append_output(self, output: Output):
        self.batch.append(output)

    def _sum_output_list(self) -> List[pd.DataFrame]:
        return [output.sum_output() for output in self.batch]

    def sum_batch(self, how: List[str] = ["summed_output_list", "mean_and_std"]):
        # sum types: list of dfs for some other part in code, concated dfs?, mean_and_std
        if "summed_output_list" in how:
            self.summed_output_list = self._sum_output_list()
        if "mean_and_std" in how:
            sol = getattr(self, "summed_output_list", self._sum_output_list())
            concated = pd.concat(sol).groupby(self.dataset.interval)
            self.mean_and_std = {
                "mean": concated.mean().to_dict(),
                "std": concated.std().to_dict(),
            }

    def load(self, file_path=None, format_="json"):
        if not file_path:
            file_path = OUTPUT_FOLDER / f"{self.task.id}.{format_}"
        if file_path.suffix == ".json":
            with open(file_path, "r") as f:
                self.mean_and_std = json.load(f)
        else:
            if file_path.suffix == ".csv":
                concated_df = pd.read_csv(file_path).set_index("day")
            elif file_path.suffix == ".pbz2":
                concated_df = self.load_pickle(file_path)
            self.summed_output_list = np.array_split(
                concated_df, self.task["ITERATIONS"]
            )

    def visualize(self):
        df = (
            pd.DataFrame(self.mean_and_std["mean"])
            .drop(columns=["infected_daily", "daily_infectors"])
            .reset_index()
            .rename(columns={"index": "day"})
            .melt(id_vars="day", var_name="color", value_name="amount")
        )
        df["day"] = df["day"].astype(int)
        return self.visualizer.stacked_bar(df)


class MultiBatch(OutputBase):
    """
    Results of all iterations with all configs

    Attributes
    ----------
    batches : Dict[str, Dict[Union[int, float], Batch]]
        {param: {step: Batch, step: ...}, ...}
        list of one Output per iteration
    summed_analysis : pd.DataFrame
        value | metric | step | parameter

    Methods
    -------
    append_batch()
    analysis_sum()
    load()
    visualize()

    Inherited
    ---------
    Attributes: task, dataset, visualizer
    Methods: export()
    """

    def __init__(self, dataset: Dataset, task: Task):
        super().__init__(dataset=dataset, task=task)
        self.batches: Dict[str, Dict[Union[int, float], Batch]] = {}
        self.analysis = Analysis(self.dataset, self.task)
        self.summed_analysis = pd.DataFrame(
            columns=["value", "metric", "step", "parameter"]
        )

    def append_batch(self, batch: Batch, param: str, step: Union[int, float]):
        batch.sum_batch()
        results = []
        for metric in self.task["sensitivity"]["metrics"]:
            if metric["grouping"] == "r_0":
                # FIXME: results += self.analysis.r_0(batch)
                pass
            else:
                results += self.analysis.group_count(batch, **metric)
        results = (
            pd.DataFrame(results, columns=["value", "metric"])
            .reset_index(drop=True)
            .assign(**{"step": step, "parameter": param})
        )
        self.summed_analysis = self.summed_analysis.append(results)
        del batch.summed_output_list
        self.batches.setdefault(param, {})[step] = batch

    def load(self, file_path=None, format_="json"):
        if not file_path:
            file_path = OUTPUT_FOLDER / f"{self.task.id}.{format_}"
        if file_path.suffix == ".csv":
            self.batches = pd.read_csv(file_path, dtype={"step": str})
        elif file_path.suffix == ".pbz2":
            self.batches = self.load_pickle(file_path)
        elif file_path.suffix == ".json":
            self.batches = {}
            with open(file_path, "r") as f:
                d = json.load(f)
            for param, d_ in d.items():
                self.batches[param] = {}
                for step, batch in d_.items():
                    self.batches[param][step] = Batch(
                        self.dataset, self.task, init=batch
                    )

    def _prep_for_vis(self, param: str) -> pd.DataFrame:
        to_concat = [
            pd.DataFrame(self.batches[param][step]["mean"]).assign(**{"step": step})
            for step in self.batches[param].keys()
        ]
        return (
            pd.concat(to_concat)
            .drop(columns=["infected_daily", "daily_infectors"])
            .reset_index()
            .rename(columns={"index": "day"})
            .melt(id_vars=["day", "step"], var_name="color", value_name="amount")
        )

    def visualize(self, how: List[str] = ["boxplots", "stacked_bars"]):
        for vis in how:
            if vis == "stacked_bars":
                for param in self.batches.keys():
                    self.visualizer.stacked_bar(self._prep_for_vis(param), param=param)
            if vis == "boxplots":
                self.summed_analysis["step"] = self.summed_analysis["step"].astype(int)
                self.visualizer.boxplots(self.summed_analysis)