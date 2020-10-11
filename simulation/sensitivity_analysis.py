import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from simulation.building_blocks import ConnectedBasicBlock
from simulation.output import Output, MultiBatch
from simulation.constants import *
from simulation.contagion import ContagionRunner


class SensitivityRunner(ConnectedBasicBlock):  # (Runner?)
    def run(self) -> MultiBatch:
        cr = ContagionRunner(self.dataset, self.task, self.gcloud)
        metabatch = MultiBatch(self.task)
        sa_conf = self.task["sensitivity"]
        for param in sa_conf["params"]:
            print(f"running sensitivity analysis on {param}")
            baseline = self.task[param]
            sr = sa_conf["ranges"][param]
            times = int((sr["max"] - sr["min"]) / sr["step"])
            for i in range(int(times) + 1):
                value = round(sr["min"] + i * sr["step"], 2)  # wierd float stuff
                print(f"checking when {param} = {value}")
                self.task.update({param: value})
                output = Output(self.dataset, self.task)
                batch = cr.run()
                step = int(round(self.task[param] - baseline, 1) / sr["step"])
                relative_steps = f"{('+' if step > 0 else '')}{step}"
                metabatch.append_batch(batch, param, value, relative_steps)
            self.task[param] = baseline
        return metabatch
