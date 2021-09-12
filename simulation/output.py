from __future__ import annotations

import gzip
import json
from os import execle
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from simulation.building_blocks import BasicBlock
from simulation.constants import *
from simulation.dataset import Dataset
from simulation.helpers import timing
from simulation.task import Task
from simulation.visualizer import Visualizer

if TYPE_CHECKING:
    from sensitivity_analysis import Analysis


class OutputBase(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task):
        super().__init__(dataset=dataset, task=task)
        self.visualizer = Visualizer(dataset, task, save=True)

    def _export_gzip(self, d: Union[Dict, pd.DataFrame], attr: str):
        with gzip.GzipFile(OUTPUT_FOLDER / f"{self.task.id}_{attr}.pgz", "w") as f:
            pickle.dump(d, f)

    def export(self, *what: str, table_format: str = "csv"):
        # TODO: clean up
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
                    try:
                        json.dump(d, fp)
                    except TypeError:
                        self._export_gzip(d, attr)
            if isinstance(d, pd.DataFrame):
                if table_format == "pickle":
                    self._export_gzip(d, attr)
                elif table_format == "csv":
                    d.to_csv(
                        OUTPUT_FOLDER / f"{self.task.id}_{attr}.csv",
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
        infection_date | days_left | color | variant
    summed : dict
    variant_summed :

    """

    def __init__(
        self,
        dataset: Dataset,
        task: Task,
        loaded: Optional[dict] = None,
    ):
        super().__init__(dataset=dataset, task=task)
        self.df = pd.DataFrame(
            [],
            columns=[
                "infection_date",
                "days_left",
                "color",
                "variant",
            ],
        ).rename_axis(index="source")
        self.df["variant"] = self.df["variant"].astype(self.variants.categories)
        self.df["color"] = self.df["color"].astype(self.states.categories("states"))
        self.summed = loaded if loaded else {}
        if self.variants.exist:
            self.variant_summed = {}  # TODO: add not_green count

    def set_first(self):
        if not hasattr(self, "first"):
            self.first = self.df[self.df["infection_date"] == 0].index[0]

    def set_damage_assessment(self):
        damage = sum(
            [
                v
                for k, v in self.summed[max(self.summed)].items()
                if k not in ({"green"} | self.states.non_states)
            ]
        )
        self.damage_assessment = [self.first, damage]

    def __len__(self):
        return len(self.df.index)

    def sum_output(self) -> pd.DataFrame:
        if self.variants.exist:
            df = pd.concat(
                [
                    pd.DataFrame(summed)
                    .T.rename_axis("day", axis="index")
                    .fillna(0)
                    .assign(variant=variant)
                    for variant, summed in self.variant_summed.items()
                ]
            )
        else:
            df = pd.DataFrame(self.summed).T.rename_axis("day", axis="index").fillna(0)
            for metric in self.states.states_with_duration:
                if metric not in df.columns:
                    df[metric] = 0
        return df

    def _regular_count(self, day: int, daily: pd.DataFrame):
        self.summed[day] = dict(self.df["color"].value_counts())
        self.summed[day]["green"] = self.dataset.nodes - sum(self.summed[day].values())
        self.summed[day]["daily_infected"] = len(daily)
        # FIXME: we stoped keeping daily_infectors, so this will always return 0
        # notna_infectors = daily[daily["infector"].notna()]["infector"]
        # self.summed[day]["daily_infectors"] = (
        #     len(set.union(*notna_infectors)) if len(notna_infectors) > 0 else 0
        # )
        self.summed[day]["daily_blue"] = len(daily[daily["color"] == "blue"])
        self.summed[day]["sick"] = len(
            self.df[self.df["color"].isin(self.states.sick_states)]
        )

    def _variant_count(self, day: int, daily: pd.DataFrame):
        summed = pd.DataFrame(
            {
                "infected": self.df.groupby("variant")["color"].count(),
                "daily_infected": daily.groupby("variant")["color"].count(),
                "sick": self.df[self.df["color"].isin(self.states.sick_states)]
                .groupby("variant")["color"]
                .count(),
            }
        ).to_dict("index")
        for k, v in summed.items():
            self.variant_summed.setdefault(k, {day: v})
            self.variant_summed[k][day] = v

    def value_counts(self, day: int):
        daily = self.df[self.df["infection_date"] == day]
        if self.variants.exist:
            self._variant_count(day, daily)
        self._regular_count(day, daily)


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

    def sum_batch(
        self,
        how: List[str] = ["summed_output_list", "mean_and_std", "damage_assessment"],
    ):
        # sum types: list of dfs for some other part in code, concated dfs?, mean_and_std
        if "summed_output_list" in how:
            self.summed_output_list = self._sum_output_list()
        if "mean_and_std" in how:
            sol = getattr(self, "summed_output_list", self._sum_output_list())
            concated = pd.concat(sol).groupby(
                ["day"] + (["variant"] if self.variants.exist else [])
            )
            self.mean_and_std = {
                "mean": concated.mean().to_dict(),
                "std": concated.std().to_dict(),
            }
        if "damage_assessment" in how:
            self.damage_assessment = pd.DataFrame(
                [o.damage_assessment for o in self.batch],
                columns=["patient_zero", "not_green"],
            )

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
        df = pd.DataFrame(self.mean_and_std["mean"])
        if self.variants.exist:
            df = (
                df.reset_index()
                .rename(columns={"level_0": "day", "level_1": "variant"})
                .melt(id_vars=["day", "variant"], var_name="color", value_name="amount")
            )
        else:
            df = (
                df.drop(columns=self.states.non_states & set(df.columns))
                .reset_index()
                .rename(columns={"index": "day"})
                .melt(id_vars="day", var_name="color", value_name="amount")
            )
            df["day"] = df["day"].astype(int)
        if self.variants.exist:
            return self.visualizer.line(df)
        else:
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
        value | metric | step | parameter | variant

    Methods
    -------
    append_batch()
    load()
    visualize()

    Inherited
    ---------
    Attributes: task, dataset, visualizer
    Methods: export()
    """

    def __init__(self, dataset: Dataset, task: Task, analysis: Analysis):
        super().__init__(dataset=dataset, task=task)
        self.batches: Dict[str, Dict[Union[int, float], Batch]] = {}
        self.summed_analysis = pd.DataFrame(
            columns=["value", "metric", "step", "parameter"]
        )
        self.analysis = analysis
        self.damage_assessment = pd.DataFrame(columns=["patient_zero", "not_green"])

    def append_batch(self, batch: Batch, param: str, step: Union[int, float]):
        batch.sum_batch()
        self.damage_assessment = self.damage_assessment.append(batch.damage_assessment)
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
        df = pd.concat(to_concat)
        # to_drop =
        return (
            df.drop(columns=self.states.non_states & set(df.columns))
            .reset_index()
            .rename(columns={"index": "day"})
            .melt(id_vars=["day", "step"], var_name="color", value_name="amount")
        )

    def visualize(self, how: List[str] = ["boxplots", "stacked_bars"]):
        for vis in how:
            if vis == "stacked_bars":
                if self.variants.exist:
                    for param in self.batches.keys():
                        self.visualizer.line(self._prep_for_vis(param), param=param)
                else:
                    for param in self.batches.keys():
                        self.visualizer.stacked_bar(
                            self._prep_for_vis(param), param=param
                        )
            if vis == "boxplots":
                # self.summed_analysis["step"] = self.summed_analysis["step"]
                self.visualizer.boxplots(self.summed_analysis)


class IterBatch:
    def __init__(self, results: Optional[dict]):
        self.results = results if results else None
        self.splits = {
            1000: 1,
            500: 2,
            333: 3,
            250: 4,
            200: 5,
            166: 6,
            142: 7,
            125: 8,
            111: 9,
            100: 10,
        }

    def get_sensitivity_results(self, dataset: Dataset, result: MultiBatch):
        self.results[dataset.name] = {
            k: d["value"].tolist()
            for k, d in result.summed_analysis[
                result.summed_analysis["metric"] == "max_percent_not_green"
            ].groupby("step")
        }

    def get_results(self, dataset: Dataset, result: Batch):
        self.results[dataset.name] = result.damage_assessment["not_green"].tolist()

    def export(self, task):
        with open(OUTPUT_FOLDER / f"iter_datasets_{task.id}.json", "w") as fp:
            json.dump(self.results, fp)

    def _to_split(self, s):
        return self.splits[int(s.split("_")[3])]

    def visualize(self, sample_dataset: Dataset, sample_task: Task):
        visualizer = Visualizer(sample_dataset, sample_task, save=True)
        df = (
            pd.DataFrame(self.results)
            .T.reset_index()
            .melt(id_vars="index")
            .rename(columns={"index": "step"})
            .drop(columns=["variable"])
        ).assign(**{"metric": "blue", "parameter": "spatial divide"})
        df["step"] = df["step"].apply(self._to_split)
        df["value"] = df["value"] / sample_dataset.nodes
        return visualizer.boxplot(df)
