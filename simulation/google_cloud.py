from google.cloud import storage, datastore
from google.api_core.exceptions import NotFound
from pathlib import Path
from datetime import datetime
import os, subprocess
import requests
import numpy as np
from typing import List, Tuple

from simulation.constants import *
from simulation.helpers import timing
from simulation.task import Task


class GoogleCloud(object):
    def __init__(self):
        self.s_client = storage.Client()
        self.ds_client = datastore.Client()
        self.todo = []
        self.done = []

    @timing
    def upload(
        self, filename: Path, new_name: str = None, bucket_name: str = "simulation_runs"
    ):
        """
        Expect `filename` to be output.csv_path, whereas `newname` without extension
        """
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
    def download(
        self,
        blob_name: str,
        destination: Path = DATA_FOLDER,
        bucket_name: str = "simulation_datasets",
    ):
        destination_path = destination / blob_name
        bucket = self.s_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        try:
            blob.reload()
        except NotFound:
            # if input("doesn't exist in cloud. should i upload? [y/N] ")
            print("file doesn't exist in cloud storage")
            return
        if not destination_path.exists():
            blob.download_to_filename(destination_path)
        elif os.path.getsize(destination_path) != blob.size:
            blob.download_to_filename(destination_path)
        else:
            # print(f"{destination_path.name} already exists")
            pass
        print(f"finished downloading {blob_name}. thank you for your patience :)")

    def get_tasklist(self):
        query = self.ds_client.query(kind="task")
        result = list(query.fetch())  # .add_filter("done", "=", done)
        self.done = [Task(t) for t in result if t["done"] == True]
        self.todo = [Task(t) for t in result if t["done"] == False]

    def add_tasks(self, tasks: Task, done: bool = False):
        entities = []
        for task in tasks:
            task_key = self.ds_client.key("task", task.id)
            entity = datastore.Entity(key=task_key)
            task["done"] = done
            if done:
                task.update(
                    {
                        "output_url": f"https://storage.cloud.google.com/simulation_runs/{task.id}.csv",
                        "task_done": datetime.now(),
                    }
                )
            entity.update(task)
            entities.append(entity)
        if len(entities) == 1:
            self.ds_client.put(entities[0])
        else:
            batch = self.ds_client.batch()
            with batch:
                self.ds_client.put_multi(entities)
        return [e.id for e in entities]

    def write_results(self, tasks: List[Task], outputs: List) -> List[int]:
        for output in outputs:
            self.upload(output.csv_path)
        return self.add_tasks(tasks, done=True)

