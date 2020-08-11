from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Optional
from simulation.constants import OUTPUT_FOLDER

if TYPE_CHECKING:
    from simulation.output import Output, Batch, MultiBatch
    from simulation.dataset import Dataset
    from simulation.task import Task


class BasicBlock(ABC):
    def __init__(self, dataset: Dataset, task: Task):
        self.dataset: Dataset = dataset
        self.task: Task = task


class OutputBasicBlock(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, output: Output):
        super().__init__(dataset=dataset, task=task)
        self.output: "Output" = output
