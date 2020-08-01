import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List

from simulation.building_blocks import OutputBasicBlock, BasicBlock
from simulation.output import Output
from simulation.constants import *
from simulation.task import Task
from simulation.dataset import Dataset
from simulation.contagion import ContagionRunner
from simulation.visualizer import Visualizer
from simulation.analysis import Analysis


class SensitivityOutput(Output):
    def __init__(self, dataset, task):
        super().__init__(dataset=dataset, task=task)
        self.results = []

    def concat_outputs(self):
        self.concated = pd.concat(self.results)


class SensitivityRunner(OutputBasicBlock):  # (Runner?)
    def __init__(self, dataset, task, output):
        super().__init__(dataset=dataset, task=task, output=output)
        self.analysis = Analysis(dataset=dataset, task=task, output=output)

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
                result = getattr(self.analysis, sa_conf["metric"])(output).assign(
                    **{"step": relative_steps, "parameter": param}
                )
                self.output.results.append(result)
            self.task[param] = baseline
        visualizer = Visualizer(
            output=self.output, task=self.task, dataset=self.dataset
        )
        visualizer.sensitivity_boxplot(grouping="step")
        return self.output
