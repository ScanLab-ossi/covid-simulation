from simulation.constants import *
from simulation.contagion import ContagionRunner
from simulation.dataset import Dataset
from simulation.dbox import Dropbox
from simulation.google_cloud import GoogleCloud
from simulation.sensitivity_analysis import SensitivityRunner
from simulation.task import Task
from simulation.output import IterBatch, MultiBatch


def main():
    gcloud, dropbox = GoogleCloud(), Dropbox()
    if settings.get("ITER_DATASET", False):
        tasklist = [
            Task({"DATASET": Path(b.name).stem})
            for b in list(gcloud.client.list_blobs("simulation_datasets"))
            if config["meta"]["DATASET"] in b.name
        ]
        iterbatch = IterBatch()
    elif settings["LOCAL_TASK"]:
        tasklist = [
            Task(path=p) for p in CONFIG_FOLDER.iterdir() if p.name.startswith("config")
        ]
    for task in tasklist:
        print(f"starting task {task.id}")
        task.export()
        task.export("poi", "print")
        # print_settings(task)
        if settings["UPLOAD"] and not settings["ITER_DATASET"]:
            dropbox.upload(task.path, task.id)
        dataset = Dataset(task["DATASET"], task=task, gcloud=gcloud)
        runner = (
            SensitivityRunner(dataset, task)
            if task["SENSITIVITY"]
            else ContagionRunner(dataset, task)
        )
        result = runner.run()
        # result.get_all_data()
        result.export()
        # result.export("metrics")
        # result.export(("batches" if isinstance(result, MultiBatch) else "batch"))
        for how in task["visualize"]["how"]:
            getattr(result, f"visualize_{how}")()
        if settings["UPLOAD"]:
            dropbox.write_results(task)
        if settings["ITER_DATASET"]:
            iterbatch.get_sensitivity_results(dataset, result) if task[
                "SENSITIVITY"
            ] else iterbatch.get_results(dataset, result)
    if settings["ITER_DATASET"]:
        iterbatch.export(tasklist[0])
        iterbatch.visualize(tasklist[0].dataset, tasklist[0])
        if settings["UPLOAD"]:
            dropbox.write_results(tasklist[0])
    return []


if __name__ == "__main__":
    main()
