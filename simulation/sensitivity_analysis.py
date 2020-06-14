import pandas as pd
from typing import List

from simulation.output import Output
from simulation.constants import *
from simulation.basic_configuration import BasicConfiguration
from simulation.task import Task
from simulation.dataset import Dataset
from simulation.contagion import ContagionRunner
from simulation.visualizer import Visualizer


class SensitivityOutput(Output):
    def __init__(self, dataset, task):
        super().__init__(dataset, task)
        self.results = []

    def concat_outputs(self):
        self.concated = pd.concat(self.results)


class SensitivityRunner(object):  # (Runner?)
    def __init__(self, dataset: Dataset, output: SensitivityOutput, task: Task):
        self.dataset = dataset
        self.output = output
        self.task = task

    def sensitivity_runner(self) -> SensitivityOutput:
        sa_conf = self.task["sensitivity"]
        for param in sa_conf["params"]:
            print(
                f"running sensitivity analysis with metric {sa_conf['metric']} on {param}"
            )
            baseline = self.task[param]
            sr = sa_conf["ranges"][param]
            times = int((sr["max"] - sr["min"]) / sr["step"])
            for i in range(int(times) + 1):
                value = round(sr["min"] + i * sr["step"], 2)  # wierd float stuff
                print(f"checking when {param} = {value}")
                self.task.update({param: value})
                output = Output(self.dataset, self.task)
                ContagionRunner.contagion_runner(
                    self.dataset, output, self.task, reproducable=False
                )
                step = int(round(self.task[param] - baseline, 1) / sr["step"])
                relative_steps = f"{('+' if step > 0 else '')}{step}"
                result = getattr(Analysis, sa_conf["metric"])(output).assign(
                    **{"step": relative_steps, "parameter": param}
                )
                self.output.results.append(result)
            self.task[param] = baseline
        visualizer = Visualizer(output=self.output, task=self.task)
        visualizer.sensitivity_boxplot(grouping="step")
        return self.output


class Analysis(object):
    @staticmethod
    def peak_sick(output: Output, what: str = "amount") -> pd.DataFrame:
        fname = "peak_sick"
        summed = [
            output.sum_output(df).pivot(index="day", columns="color")["amount"]
            for df in output.batch
        ]
        bpr = [df["b"] + df["p"] + df["r"] for df in summed]
        if what == "amount":
            return pd.DataFrame([df.max() for df in bpr], columns=[fname])
        elif what == "day":
            return pd.DataFrame([df.idxmax() for df in bpr], columns=[fname])

    @staticmethod
    def peak_newly_infected(output: Output, what: str = "amount") -> pd.DataFrame:
        fname = "peak_newly_infected"
        grouped = [
            df.groupby("infection_date")["final_state"].count().reset_index(drop=True)
            for df in output.batch
        ]
        if what == "amount":
            return pd.DataFrame([df.max() for df in grouped], columns=[fname])
        elif what == "day":
            return pd.DataFrame([df.idxmax() for df in grouped], columns=[fname])
