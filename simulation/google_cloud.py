from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Dict, List

import requests
from cachetools import LFUCache, cached
from google.api_core.exceptions import NotFound  # type: ignore
from google.cloud import secretmanager_v1, storage  # type: ignore

from simulation.constants import *
from simulation.helpers import timing

cache = LFUCache(1000)


class GoogleCloud:
    def __init__(self):
        self.add_keyfile()
        self.client = storage.Client()
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
        bucket = self.client.bucket(bucket_name)
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
        destination_path = destination / blob_name
        if destination_path.exists():
            if settings["VERBOSE"]:
                print(f"skipped downloading {destination_path.name}, since it already exists")
                return
        else:
            bucket = self.client.bucket(bucket_name)
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