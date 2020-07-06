import json, os, sys
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from typing import Union
from flask import Flask, jsonify

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

# TODO:
# async?


def main(test_conf: dict = False):
    basic_conf = BasicConfiguration()
    gcloud = GoogleCloud(basic_conf)
    if settings["LOCAL_TASK"]:
        tasklist = [Task()]
    else:
        gcloud.get_tasklist()
        tasklist = gcloud.todo
        if len(tasklist) == 0:
            print("you've picked LOCAL_TASK=False, but no tasks are waiting")
            return []
    tasks, outputs = [], []
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
        tasks.append(task)
        outputs.append(output)
    if settings["UPLOAD"]:
        return gcloud.write_results(tasks, outputs)


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

