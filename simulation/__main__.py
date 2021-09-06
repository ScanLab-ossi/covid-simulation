import json

from constants import *
from contagion import ContagionRunner
from dataset import Dataset
from dbox import Dropbox
from google_cloud import GoogleCloud
from helpers import print_settings
from sensitivity_analysis import SensitivityRunner
from task import Task


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
        tasklist = [
            Task(path=p) for p in CONFIG_FOLDER.iterdir() if p.name.startswith("config")
        ]
    for task in tasklist:
        print(f"starting task {task.id}")
        print_settings(task)
        if settings["UPLOAD"] and not settings["ITER_DATASET"]:
            dropbox.upload(task.path, task.id)
            # FIXME: new split config
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
            if task["SENSITIVITY"]:
                iter_results[dataset.name] = {
                    k: d["value"].tolist()
                    for k, d in result.summed_analysis[
                        result.summed_analysis["metric"] == "max_percent_not_green"
                    ].groupby("step")
                }
            else:
                iter_results[dataset.name] = result.damage_assessment[
                    "not_green"
                ].tolist()
    if settings["ITER_DATASET"]:
        with open(OUTPUT_FOLDER / f"iter_datasets_{tasklist[0].id}.json", "w") as fp:
            json.dump(iter_results, fp)
        if settings["UPLOAD"]:
            dropbox.write_results(tasklist[0])
    return []


if __name__ == "__main__":
    main()
