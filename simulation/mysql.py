from pathlib import Path
import configparser, os

import pandas as pd
from sqlalchemy import create_engine, engine

from simulation.constants import *
from simulation.helpers import timing
from simulation.google_cloud import GoogleCloud


class MySQL:
    def __init__(self, gcloud: GoogleCloud):
        self.gcloud = gcloud
        self.config = self.get_config()
        self.engine = self.get_engine()

    def get_config(self) -> configparser.ConfigParser:
        conf_file = Path("./secrets.conf")
        conf = configparser.ConfigParser()
        if not conf_file.exists():
            env_vars = [
                "MYSQL_USER",
                "MYSQL_PASSWORD",
                "MYSQL_HOST",
                "MYSQL_CLOUDSQL_INSTANCE",
            ]
            try:
                conf["mysql"] = self.gcloud.get_secrets(env_vars)
            except Exception as e:
                print(e)
                conf["mysql"] = {k: os.environ[k] for k in env_vars}
        else:
            conf.read(conf_file)
        res = {k.lower(): v for k, v in conf.items()}
        print(res)
        return res

    def get_engine(self) -> engine.Engine:
        CONN_STR = "mysql+mysqldb://{mysql_user}:{mysql_password}@{mysql_host}/datasets?unix_socket=/cloudsql/{mysql_cloudsql_instance}".format(
            **dict(self.config["mysql"])
        )
        return create_engine(CONN_STR, echo=False, pool_size=1, max_overflow=0)

    @timing
    def query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.engine)
