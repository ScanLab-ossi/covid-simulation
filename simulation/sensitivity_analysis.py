from simulation.building_blocks import ConnectedBasicBlock
from simulation.constants import *
from simulation.contagion import ContagionRunner
from simulation.output import MultiBatch


class SensitivityRunner(ConnectedBasicBlock):  # (Runner?)
    def _times(self, sr: dict) -> int:
        return int((sr["max"] - sr["min"]) / sr["step"])

    def _steps(self, value: float, baseline: float, step: float) -> str:
        step = int(round((value - baseline) / step, 1))
        return f"{('+' if step > 0 else '')}{step}"

    def run(self) -> MultiBatch:
        cr = ContagionRunner(self.dataset, self.task, self.states, self.gcloud)
        multibatch = MultiBatch(self.dataset, self.task, self.states)
        sa_conf = self.task["sensitivity"]
        for param in sa_conf["params"]:
            print(f"running sensitivity analysis on {param}")
            if param not in self.task.keys():
                sub = [k for k in self.task["paths"][param].keys() if k[0] == "d"][0]
                baseline = self.task["paths"][param][sub]
                sr = sa_conf["ranges"][param]
                times = self._times(sr)
                for i in range(times + 1):
                    v = round(sr["min"] + i * sr["step"], 2)
                    value = (
                        [v, baseline[1]] if sub == "duration" else [v, round(1 - v, 2)]
                    )
                    print(f"checking when {param} {sub} = {value}")
                    self.task["paths"][param].update({sub: value})
                    batch = cr.run()
                    multibatch.append_batch(
                        batch=batch, param=f"{param}__{sub}", step=v
                    )  # self._steps(v, baseline[0], sr["step"])
                self.task["paths"][param][sub] = baseline
            else:
                baseline = self.task[param]
                sr = sa_conf["ranges"][param]
                times = self._times(sr)
                for i in range(times + 1):
                    value = round(sr["min"] + i * sr["step"], 2)  # wierd float stuff
                    print(f"checking when {param} = {value}")
                    self.task.update({param: value})
                    batch = cr.run()
                    multibatch.append_batch(batch=batch, param=param, step=value)
                self.task[param] = baseline
        return multibatch
