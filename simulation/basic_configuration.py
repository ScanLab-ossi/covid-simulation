from pathlib import Path
import configparser, os, json


class BasicConfiguration(object):
    """
    Initialize all *technical* configuration aspects (postgres, google)
    This is not the configuration for the simulation run (called that task_conf)
    """

    def __init__(self):
        self.add_keyfile()
        self.config = self.get_config()

    def get_config(self):
        conf_file = Path("./secrets.conf")
        conf = configparser.ConfigParser()
        if not conf_file.exists():
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
        if not Path("./keyfile.json").exists():
            with open("keyfile.json", "w") as fp:
                json.dump(json.loads(os.environ["KEYFILE"]), fp)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"
