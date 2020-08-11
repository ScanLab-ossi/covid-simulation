import json, os, sys
from datetime import datetime, date, timedelta
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pathlib import Path
import multiprocessing as mp
from typing import Union
from flask import Flask, jsonify

from simulation.helpers import timing, one_array_pickle_to_set, print_settings
from simulation.constants import *
from simulation.task import Task
from simulation.google_cloud import GoogleCloud
from simulation.dataset import Dataset
from simulation.contagion import ContagionRunner
from simulation.output import Output
from simulation.sensitivity_analysis import SensitivityRunner
from simulation.visualizer import Visualizer
from simulation.analysis import Analysis


def main():
    gcloud = GoogleCloud()
    if settings["LOCAL_TASK"]:
        tasklist = [Task()]
    else:
        gcloud.get_tasklist()
        tasklist = gcloud.todo
        if len(tasklist) == 0:
            print("you've picked LOCAL_TASK=False, but no tasks are waiting")
            return []
    tasks, results = [], []
    for task in tasklist:
        print(f"starting task {task.id}")
        print_settings()
        dataset = Dataset(task["DATASET"])
        dataset.load_dataset(gcloud=gcloud)
        runner = (
            SensitivityRunner(dataset, task)
            if task["SENSITIVITY"]
            else ContagionRunner(dataset, task)
        )
        result = runner.run()
        if task["SENSITIVITY"]:
            analysis = Analysis(dataset, task)
            result.analysis_sum(analysis)
            result.export()
        else:
            result.sum_all_and_concat()
            result.export()
        # if settings["LOCAL"]:
        #     visualizer = Visualizer(
        #         task=task, dataset=dataset, batches=result, save=True
        #     )
        #     if task["SENSITIVITY"]:
        #         visualizer.sensitivity_boxplot()
        #     else:
        #         visualizer.visualize()
        results.append(result)
        tasks.append(task)
    if settings["UPLOAD"]:
        return gcloud.write_results(tasks, results)


# if settings["PARALLEL"]:
#     with mp.Pool() as p:
#         r = [
#             p.apply_async(
#                 ContagionRunner.contagion_runner, (dataset, output, task),
#             )
#             for _ in range(task["ITERATIONS"])
#         ]
#         output.concated = pd.concat([res.get() for res in r])
#         output.export(how="average")


app = Flask(__name__)


@app.route("/")
def run_main():
    tasks_finished = main()
    return jsonify(tasks_finished)


if __name__ == "__main__":
    if settings["LOCAL"] == True:
        main()
    else:
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

