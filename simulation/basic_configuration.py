from pathlib import Path
import configparser, os, json
from simulation.constants import *

from google.cloud import secretmanager_v1


class BasicConfiguration(object):
    """
    Initialize all *technical* configuration aspects (postgres, google)
    This is not the configuration for the simulation run (called that task_conf)
    """

    def __init__(self):
        self.add_keyfile()
        self.config = self.get_config()
        self.client = secretmanager_v1.SecretManagerServiceClient()

    def get_secret(self, secret_name):
        name = self.client.secret_version_path(
            "temporal-dynamics", secret_name, "latest"
        )
        response = self.client.access_secret_version(name)
        return response.payload.data.decode("utf-8")

    def get_config(self):
        conf_file = Path("./secrets.conf")
        conf = configparser.ConfigParser()
        if not conf_file.exists():
            try:
                conf["postgres"] = {
                    "dbname": self.get_secret("POSTGRES_DATABASE_NAME"),
                    "user": self.get_secret("POSTGRES_USER_NAME"),
                    "password": self.get_secret("POSTGRES_USER_PASSWORD"),
                    "host": self.get_secret("POSTGRES_DATABASE_HOST"),
                }
            except Exception:
                conf["postgres"] = {
                    "dbname": os.environ["POSTGRES_DATABASE_NAME"],
                    "user": os.environ["POSTGRES_USER_NAME"],
                    "password": os.environ["POSTGRES_USER_PASSWORD"],
                    "host": os.environ["POSTGRES_DATABASE_HOST"],
                }
        else:
            conf.read(conf_file)
        return conf

    def add_keyfile(self):
        if settings["LOCAL"]:
            if not Path("./keyfile.json").exists():
                with open("keyfile.json", "w") as fp:
                    json.dump(json.loads(os.environ["KEYFILE"]), fp)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"
