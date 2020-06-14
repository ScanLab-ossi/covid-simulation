from google.cloud import storage, datastore
from google.api_core.exceptions import NotFound
from pathlib import Path
from datetime import datetime
import os, subprocess
import requests
import numpy as np

from simulation.constants import *
from simulation.helpers import timing
from simulation.basic_configuration import BasicConfiguration
from simulation.task import Task


class GoogleCloud(object):
    def __init__(self, config: BasicConfiguration):
        self.config = config
        self.s_client = storage.Client()
        self.ds_client = datastore.Client()
        self.todo = []
        self.done = []

    @timing
    def upload(
        self, filename: Path, new_name: str = None, bucket_name: str = "simulation_runs"
    ):
        then = datetime.now()
        bucket = self.s_client.bucket(bucket_name)
        blob_name = f"{new_name}.csv" if new_name else filename.name
        blob = bucket.blob(blob_name)
        with open(filename, "rb") as f:
            file_size = os.path.getsize(filename)
            if file_size < 10_485_760:  # 10MB
                blob.upload_from_file(f, content_type="text/csv")
            else:
                url = blob.create_resumable_upload_session(
                    content_type="text/csv", size=file_size
                )
                res = requests.put(url, data=f)
                res.raise_for_status()
        print(f"uploaded {blob_name} to {bucket.name}. took {datetime.now() - then}")
        return blob.self_link

    # @timing
    def download(self, blob_name: str):
        destination_path = Path(DATA_FOLDER / blob_name)
        bucket = self.s_client.bucket("simulation_datasets")
        blob = bucket.blob(blob_name)
        try:
            blob.reload()
        except NotFound:
            # if input("doesn't exist in cloud. should i upload? [y/N] ")
            print("file doesn't exist in cloud storage")
            return
        if (
            not destination_path.exists()
            or os.path.getsize(destination_path) != blob.size
        ):
            print(f"downloading {blob_name}. this might take a while.")
            blob.download_to_filename(destination_path)
            print(f"finished downloading {blob_name}. thank you for your patience :)")
        else:
            pass
            # print(f"{destination_path.name} already exists")

    def get_tasklist(self, done=False):
        query = self.ds_client.query(kind="task")
        result = list(query.fetch())  # .add_filter("done", "=", done)
        self.done = [Task(t) for t in result if t["done"] == True]
        self.todo = [Task(t) for t in result if t["done"] == False]

    def add_task(self, task: Task, done=False):
        if done:
            task_key = self.ds_client.key("task", task.id)
        else:
            task_key = self.ds_client.key("task")
        g_task = datastore.Entity(key=task_key)
        task["done"] = done
        g_task.update(task)
        if done:
            task.update(
                {
                    "output_url": f"https://storage.cloud.google.com/simulation_runs/{task_key.id}.csv",
                    "task_done": datetime.now(),
                }
            )
            g_task.update(task)
        self.ds_client.put(g_task)
        return task_key.id

    def write_results(self, result):
        if len(result) > 1:
            batch = self.ds_client.batch()
            with batch:
                tasks = []
                for output, task in result:
                    link = self.upload(output.csv_path, new_name=task.id)
                    task.update(
                        {"done": True, "task_done": datetime.now(), "link": link}
                    )
                    tasks.append(dict(task))
                self.ds_client.put_multi(tasks)
        else:
            data, task = result[0]
            link = self.upload(data.output_filename, new_name=task.id)
            task.update({"done": True, "task_done": datetime.now(), "link": link})
            self.ds_client.put(task)
