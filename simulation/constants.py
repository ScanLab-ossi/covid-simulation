from pathlib import Path
import os

DATA_FOLDER = Path("./data").resolve()
OUTPUT_FOLDER = Path("./output").resolve()
# PARENT_FOLDER = Path("./simulation").resolve()

# using os.environ for flexibility in the CI

# which dataset to use - TODO: move to different configuration
DATASET = os.environ.get("DATASET", "copenhagen_interactions")

# run experiment X times, so to average chaotic(?) effects.
# use 1 repetition for working on the model itself
REPETITIONS = os.environ.get("REPETITIONS", 1)

# run simulation in parallel, rather than iteratively.
# iterative seems to run faster locally
PARALLEL = os.environ.get("PARALLEL", False)

# task list according to local config onl.
# skips downloading task list from google cloud
LOCAL = os.environ.get("LOCAL", True)

# uplaod results to google cloud
UPLOAD = os.environ.get("UPLOAD", False)

# maximum prints, for testing withuor repetitions
VERBOSE = os.environ.get("VERBOSE", (True if REPETITIONS == 1 else False))
# VERBOSE = True
# skip heavy tests - google cloud and sql
SKIP_TESTS = os.environ.get("SKIP_TESTS", True)
