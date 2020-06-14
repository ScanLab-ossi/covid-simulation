import json, os, sys
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from typing import Union

from simulation.helpers import timing, one_array_pickle_to_set, print_settings
from simulation.constants import *
from simulation.basic_configuration import BasicConfiguration
from simulation.task import Task
from simulation.google_cloud import GoogleCloud
from simulation.dataset import Dataset
from simulation.contagion import ContagionRunner
from simulation.output import Output
from simulation.sensitivity_analysis import (
    Analysis,
    SensitivityOutput,
    SensitivityRunner,
)
from simulation.visualizer import Visualizer


def main(test_conf: dict = False):
    basic_conf = BasicConfiguration()
    gcloud = GoogleCloud(basic_conf)
    if not settings["LOCAL"]:
        gcloud.get_tasklist()
        if not gcloud.todo:
            print("you've picked LOCAL=False, but no tasks are waiting")
            sys.exit()
    results = []
    tasklist = [Task()] if settings["LOCAL"] else gcloud.todo
    for task in tasklist:
        print(f"starting task {task.id}")
        print_settings()
        dataset = Dataset(task["DATASET"])
        dataset.load_dataset(gcloud=gcloud)
        output = (
            SensitivityOutput(dataset, task)
            if task["SENSITIVITY"]
            else Output(dataset, task)
        )
        if task["SENSITIVITY"]:
            sr = SensitivityRunner(dataset, output, task)
            sr.sensitivity_runner()
            output.export(how="concated")
        else:
            if settings["PARALLEL"]:
                with mp.Pool() as p:
                    r = [
                        p.apply_async(
                            ContagionRunner.contagion_runner, (dataset, output, task),
                        )
                        for _ in range(task["ITERATIONS"])
                    ]
                    output.concated = pd.concat([res.get() for res in r])
                    output.export()
            else:
                ContagionRunner.contagion_runner(dataset, output, task)
                output.sum_and_concat_outputs()
                output.average_outputs()
                output.export(filename=(str(task.id)))
                visualizer = Visualizer(output, task)
                visualizer.visualize()
                if task["ITERATIONS"] > 1:
                    visualizer.variance_boxplot()
        if settings["UPLOAD"]:
            if len(tasklist) > 1:
                results.append((output, task))
                continue
            task_key = gcloud.add_task(task, done=True)
            gcloud.upload(output.csv_path, new_name=task_key)
    if len(tasklist) > 1 and settings["UPLOAD"]:
        gcloud.write_results(results)


if __name__ == "__main__":
    main()
