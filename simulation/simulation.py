import json
import os

from flask import Flask, jsonify

from simulation.constants import *
from simulation.contagion import ContagionRunner
from simulation.dataset import Dataset
from simulation.dropbox import Dropbox
from simulation.google_cloud import GoogleCloud
from simulation.helpers import print_settings
from simulation.sensitivity_analysis import SensitivityRunner
from simulation.task import Task


def main():
    gcloud, dropbox = GoogleCloud(), Dropbox()
    if settings.get("ITER_DATASET", False):
        tasklist = [
            Task({"DATASET": Path(b.name).stem})
            for b in list(gcloud.s_client.list_blobs("simulation_datasets"))
            if config["meta"]["DATASET"] in b.name
        ]
        iter_results = {}
    elif settings["LOCAL_TASK"]:
        tasklist = [Task(path=p) for p in CONFIG_FOLDER.iterdir() if "config" in p.name]
    else:
        gcloud.get_tasklist()
        tasklist = gcloud.todo
        if len(tasklist) == 0:
            print("you've picked LOCAL_TASK=False, but no tasks are waiting")
            return []
    for task in tasklist:
        print(f"starting task {task.id}")
        print_settings(task)
        if settings["UPLOAD"] and not settings["ITER_DATASET"]:
            dropbox.upload(task.path, task.id)
        dataset = Dataset(task["DATASET"], task=task, gcloud=gcloud)
        runner = (
            SensitivityRunner(dataset, task)
            if task["SENSITIVITY"]
            else ContagionRunner(dataset, task)
        )
        result = runner.run()
        if task["SENSITIVITY"]:
            result.export("batches", "summed_analysis", "damage_assessment")
            result.visualize()  # how=[]
        else:
            result.sum_batch()
            if not settings["ITER_DATASET"]:
                result.export("mean_and_std", "damage_assessment")
                result.visualize()
        if settings["UPLOAD"]:
            dropbox.write_results(task)
        if settings["ITER_DATASET"]:
            iter_results[dataset.name] = result.damage_assessment["not_green"].tolist()
    if settings["ITER_DATASET"]:
        with open(OUTPUT_FOLDER / f"iter_datasets_{tasklist[0].id}.json", "w") as fp:
            json.dump(iter_results, fp)
        if settings["UPLOAD"]:
            dropbox.write_results(tasklist[0])
    return []
    # return gcloud.write_results(tasks)


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
