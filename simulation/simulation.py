import os

from flask import Flask, jsonify

from simulation.constants import *
from simulation.contagion import ContagionRunner
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.dropbox import Dropbox
from simulation.helpers import print_settings
from simulation.sensitivity_analysis import SensitivityRunner
from simulation.task import Task


def main():
    gcloud, dropbox = GoogleCloud(), Dropbox()
    if settings["LOCAL_TASK"]:
        tasklist = [Task()]
    else:
        gcloud.get_tasklist()
        tasklist = gcloud.todo
        if len(tasklist) == 0:
            print("you've picked LOCAL_TASK=False, but no tasks are waiting")
            return []
    tasks = []
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
            result.export("batches", "summed_analysis")
            result.visualize()
        else:
            result.sum_batch()
            result.export("mean_and_std")
            result.visualize()
        tasks.append(task)
    if settings["UPLOAD"]:
        dropbox.write_results(tasks)
        return
        # return gcloud.write_results(tasks)


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
