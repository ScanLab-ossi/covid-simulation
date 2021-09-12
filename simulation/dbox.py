import configparser
from datetime import datetime
from glob import glob
from pathlib import Path

import dropbox

from simulation.constants import *
from simulation.task import Task


class Dropbox:
    def __init__(self):
        self.dbx = dropbox.Dropbox(self.get_token())

    def get_token(self) -> str:
        conf_file = Path("./secrets.conf")
        conf = configparser.ConfigParser()
        if not conf_file.exists():
            raise FileNotFoundError
        else:
            conf.read(conf_file)
        return conf["dropbox"]["TOKEN"]

    def upload(self, source: Path, id: int):
        base_path = "/Paper- copenhagen/Results of experiments"
        today = datetime.now().strftime("%d-%m-%Y")
        subfolders = [x.name for x in self.dbx.files_list_folder(base_path).entries]
        if today not in subfolders:
            self.dbx.files_create_folder_v2(f"{base_path}/{today}")
        with open(source, "rb") as f:
            dest = (
                f"{(str(id) + '_' if str(id) not in source.name else '')}{source.name}"
            )
            self.dbx.files_upload(f.read(), f"{base_path}/{today}/{dest}")
        print("wrote to dropbox")

    def write_results(self, task: Task):
        for f in glob(str(OUTPUT_FOLDER / f"{task.id}*.html")):
            self.upload(Path(f), task.id)
        for f in glob(str(OUTPUT_FOLDER / f"{task.id}*.csv")):
            self.upload(Path(f), task.id)
        for f in glob(str(OUTPUT_FOLDER / f"iter_datasets_{task.id}*.json")):
            self.upload(Path(f), task.id)
