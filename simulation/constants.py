from pathlib import Path
import os
from yaml import load, Loader

DATA_FOLDER = Path("./data").resolve()
OUTPUT_FOLDER = Path("./output").resolve()
CONFIG_FOLDER = Path("./config").resolve()
TEST_FOLDER = Path("./tests").resolve()

settings = {}
for filename in ("default_config", "config"):
    try:
        with open(CONFIG_FOLDER / f"{filename}.yaml") as f:
            config = load(f, Loader=Loader)
            for k, v in config.get("settings", {}).items():
                settings[k] = v
    except FileNotFoundError:
        pass
settings["LOCAL"] = eval(os.environ.get("LOCAL", "True"))
try:
    settings["LOCAL_TASK"] = eval(os.environ["LOCAL_TASK"])
    settings["UPLOAD"] = eval(os.environ["UPLOAD"])
except KeyError:
    pass
