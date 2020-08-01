from pathlib import Path
import configparser, os, json
from simulation.constants import *
from simulation.google_cloud import GoogleCloud


class Postgres(object):
    def __init__(self, gcloud: GoogleCloud):
        self.gcloud = gcloud
        self.config = self.get_config()

    def get_config(self) -> configparser.ConfigParser:
        conf_file = Path("./secrets.conf")
        conf = configparser.ConfigParser()
        if not conf_file.exists():
            env_vars = [
                "POSTGRES_DATABASE_NAME",
                "POSTGRES_USER_NAME",
                "POSTGRES_USER_PASSWORD",
                "POSTGRES_DATABASE_HOST",
            ]
            try:
                conf["postgres"] = self.gcloud.get_secrets(env_vars)
            except Exception as e:
                print(e)
                conf["postgres"] = {k: os.environ[k] for k in env_vars}
        else:
            conf.read(conf_file)
        return conf

