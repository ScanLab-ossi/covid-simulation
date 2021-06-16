from pathlib import Path
import os
from yaml import load, Loader

DATA_FOLDER = Path("./data").resolve()
OUTPUT_FOLDER = Path("./output").resolve()
CONFIG_FOLDER = Path("./config").resolve()
TEST_FOLDER = Path("./tests").resolve()
# PARENT_FOLDER = Path("./simulation").resolve()

with open([p for p in CONFIG_FOLDER.iterdir() if "config" in p.name][0]) as f:
    config = load(f, Loader=Loader)
settings = config["settings"]
settings["LOCAL"] = eval(os.environ.get("LOCAL", "True"))
try:
    settings["LOCAL_TASK"] = eval(os.environ["LOCAL_TASK"])
    settings["UPLOAD"] = eval(os.environ["UPLOAD"])
except KeyError:
    pass