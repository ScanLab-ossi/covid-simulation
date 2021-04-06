from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING, Optional
from simulation.constants import OUTPUT_FOLDER

if TYPE_CHECKING:
    from simulation.output import Output, Batch
    from simulation.dataset import Dataset
    from simulation.task import Task
    from simulation.google_cloud import GoogleCloud


class BasicBlock(ABC):
    def __init__(self, dataset: Dataset, task: Task):
        self.dataset: Dataset = dataset
        self.task: Task = task


class ConnectedBasicBlock(ABC):
    def __init__(self, dataset: Dataset, task: Task, gcloud: GoogleCloud):
        self.dataset: Dataset = dataset
        self.task: Task = task
        self.gcloud = gcloud


class OutputBasicBlock(BasicBlock):
    def __init__(self, dataset: Dataset, task: Task, output: Output):
        super().__init__(dataset=dataset, task=task)
        self.output: "Output" = output
