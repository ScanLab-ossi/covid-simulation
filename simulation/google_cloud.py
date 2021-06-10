from __future__ import annotations
from datetime import datetime
import os, subprocess, json, mimetypes
from typing import List, Tuple, Dict, Union
from pathlib import Path
from glob import glob

from google.cloud import storage, datastore, secretmanager_v1  # type: ignore
from google.api_core.exceptions import NotFound  # type: ignore
import requests
from cachetools import cached, LFUCache
import numpy as np  # type: ignore

from simulation.constants import *
from simulation.helpers import timing
from simulation.task import Task
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.output import Batch, MultiBatch

cache = LFUCache(1000)


class GoogleCloud:
    def __init__(self):
        self.add_keyfile()
        self.s_client = storage.Client()
        self.ds_client = datastore.Client()
        # self.bq_client = bigquery.Client()
        self.todo = []
        self.done = []

    def add_keyfile(self):
        if settings["LOCAL"]:
            if not Path("./keyfile.json").exists():
                with open("keyfile.json", "w") as fp:
                    json.dump(json.loads(os.environ["KEYFILE"]), fp)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"

    @cached(cache)
    def get_secrets(self, secrets: List[str]) -> Dict[str, str]:
        secret_client = secretmanager_v1.SecretManagerServiceClient()
        res = {}
        for secret_name in secrets:
            name = secret_client.secret_version_path(
                "temporal-dynamics", secret_name, "latest"
            )
            response = secret_client.access_secret_version(name)
            res[secret_name] = response.payload.data.decode("utf-8")
        return res

    @timing
    def upload(
        self, filename: Path, new_name: str = None, bucket_name: str = "simulation_runs"
    ):
        bucket = self.s_client.bucket(bucket_name)
        blob_name = f"{new_name}.csv" if new_name else filename.name
        blob = bucket.blob(blob_name)
        mimetype = mimetypes.guess_type(filename)[0]
        with open(filename, "rb") as f:
            file_size = os.path.getsize(filename)
            if file_size < 10_485_760:  # 10MB
                blob.upload_from_file(f, content_type=mimetype)
            else:
                url = blob.create_resumable_upload_session(
                    content_type=mimetype, size=file_size
                )
                res = requests.put(url, data=f)
                res.raise_for_status()
        print(f"uploaded {blob_name} to {bucket.name}")
        return blob.self_link

    @timing
    def download(
        self,
        blob_name: str,
        destination: Path = DATA_FOLDER,
        bucket_name: str = "simulation_datasets",
        clean: bool = True,
    ):
        bucket = self.s_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        try:
            blob.reload()
        except NotFound:
            # if input("doesn't exist in cloud. should i upload? [y/N] ")
            if settings["VERBOSE"]:
                print("file doesn't exist in cloud storage")
                raise FileNotFoundError
            return
        if clean:
            blob_name = blob_name.split("/")[-1]
        destination_path = destination / blob_name
        if not destination_path.exists():
            blob.download_to_filename(destination_path)
            if settings["VERBOSE"]:
                print(
                    f"finished downloading {blob_name}. thank you for your patience :)"
                )
        elif os.path.getsize(destination_path) != blob.size:
            blob.download_to_filename(destination_path)
            if settings["VERBOSE"]:
                print(
                    f"finished downloading a newer version of {blob_name}. \
                        thank you for your patience :)"
                )
        else:
            if settings["VERBOSE"]:
                print(
                    f"skipped downloading {destination_path.name}, since it already exists"
                )

    def get_filelist(self, bucket_name: str = "simulation_datasets"):
        bucket = self.s_client.bucket(bucket_name)
        all_blobs = list(bucket.list_blobs(prefix="state_transitions/"))
        return [blob.name[18:-4] for blob in all_blobs if len(blob.name) > 18]

    def get_tasklist(self):
        query = self.ds_client.query(kind="task")
        result = list(query.fetch())  # .add_filter("done", "=", done)
        self.done = [Task(t) for t in result if t["done"] == True]
        self.todo = [Task(t) for t in result if t["done"] == False]

    def add_tasks(self, tasks: List[Task], done: bool = False):
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
        batch = self.ds_client.batch()
        with batch:
            self.ds_client.put_multi(entities)
        print([e.id for e in entities])
        return [e.id for e in entities]

    def write_results(self, tasks: List[Task]) -> List[int]:
        for task in tasks:
            for f in glob(str(OUTPUT_FOLDER / f"{task.id}*")):
                if "html" in f:
                    self.upload(f, bucket_name="simulation_runs/visualizations")
                else:
                    self.upload(f)
        return self.add_tasks(tasks, done=True)

    # @cached(cache)
    # def query(self, query: str) -> pd.DataFrame:
    #     query_job = self.bq_client.query(query)
    #     res = query_job.to_dataframe()
    #     return res  # Waits for job to complete.
