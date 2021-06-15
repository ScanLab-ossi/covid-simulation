from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from numpy import random

from simulation.states import States
from simulation.google_cloud import GoogleCloud

if TYPE_CHECKING:
    from simulation.dataset import Dataset
    from simulation.google_cloud import GoogleCloud
    from simulation.output import Output
    from simulation.task import Task


class BasicBlock(ABC):
    def __init__(self, dataset: Dataset, task: Task):
        self.dataset: Dataset = dataset
        self.task: Task = task
        if self.task["DATASET"] != self.dataset.name:
            raise
        self.states = States(task)  #


class RandomBasicBlock(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, reproducible: bool = False):
        super().__init__(dataset=dataset, task=task)
        self.rng = random.default_rng(42 if reproducible else None)


class ConnectedBasicBlock(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task):
        super().__init__(dataset=dataset, task=task)
        self.gcloud = GoogleCloud()


class OutputBasicBlock(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, output: Output):
        super().__init__(dataset=dataset, task=task)
        self.output: "Output" = output
