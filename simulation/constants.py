from pathlib import Path
import os
from yaml import load, Loader

DATA_FOLDER = Path("./data").resolve()
OUTPUT_FOLDER = Path("./output").resolve()
CONFIG_FOLDER = Path("./config").resolve()
TEST_FOLDER = Path("./tests").resolve()
# PARENT_FOLDER = Path("./simulation").resolve()

with open(CONFIG_FOLDER / "config.yaml") as f:
    config = load(f, Loader=Loader)
    settings = config["settings"]
    meta = config["meta"]
