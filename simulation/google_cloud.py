from google.cloud import storage, datastore
from pathlib import Path
from datetime import datetime
import os, subprocess
import requests
import numpy as np

from simulation.constants import *
from simulation.helpers import timing
from simulation.basic_configuration import BasicConfiguration
from simulation.task_config import TaskConfig


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

    @timing
    def download(self, blob_name: str):
        destination_path = Path(DATA_FOLDER / blob_name)
        if not destination_path.exists():
            bucket = self.s_client.bucket("simulation_datasets")
            blob = bucket.blob(blob_name)
            print(f"downloading {blob_name}. this might take a while.")
            blob.download_to_filename(destination_path)
            print(f"finished downloading {blob_name}. thank you for your patience :)")
        else:
            print(f"{destination_path.name} already exists")

    def get_tasklist(self, done=False):
        query = self.ds_client.query(kind="task")
        result = list(query.fetch())  # .add_filter("done", "=", done)
        self.done = [t for t in result if t["done"] == True]
        self.todo = [t for t in result if t["done"] == False]

    def add_task(self, dataset: str, task_config: TaskConfig, done=False):
        if done:
            task_key = self.ds_client.key("task", np.random.randint(1e15, 1e16))
        else:
            task_key = self.ds_client.key("task")
        task = datastore.Entity(key=task_key)
        os.chdir(Path("./simulation"))
        machine_version = (
            subprocess.check_output(
                ['git log -1 --pretty="%h" contagion.py'], shell=True
            )
            .strip()
            .decode("utf-8")
        )
        os.chdir(Path(os.getcwd()).parent)
        task.update(
            {
                "dataset": dataset,
                "config": task_config.as_lists(),
                "machine_version": machine_version,
                "task_added": datetime.now(),
                "done": done,
            }
        )
        if done:
            task.update(
                {
                    "output_url": f"https://storage.cloud.google.com/simulation_runs/{task_key}.csv",
                    "task_done": datetime.now(),
                }
            )
        self.ds_client.put(task)
        return task_key.id

    def write_results(self, result):
        if len(result) > 1:
            batch = self.ds_client.batch()
            with batch:
                tasks = []
                for data, task in result:
                    link = self.upload(data.output_filename, new_name=task.id)
                    task.update(
                        {"done": True, "task_done": datetime.now(), "link": link}
                    )
                    tasks.append(task)
                self.ds_client.put_multi(tasks)
        else:
            data, task = result[0]
            link = self.upload(data.output_filename, new_name=task.id)
            task.update({"done": True, "task_done": datetime.now(), "link": link})
            self.ds_client.put(task)
