from __future__ import annotations
from simulation.dataset import Dataset
from simulation.task import Task
from typing import Union
from abc import ABC


class BasicBlock(ABC):
    def __init__(self, dataset: Dataset, task: Task):
        self.dataset: Dataset = dataset
        self.task: Task = task


class OutputBasicBlock(BasicBlock):
    def __init__(
        self, dataset: Dataset, task: Task, output: Union["SensitivityOutput", "Output"]
    ):
        super().__init__(dataset=dataset, task=task)
        self.output: "Output" = output
