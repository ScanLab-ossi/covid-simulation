import unittest
import numpy as np
from jsonschema import Draft7Validator, validators, validate
from yaml import load, Loader
from pprint import pprint

from simulation.task import Task
from simulation.analysis import Analysis
from simulation.constants import *


metadata_keys = [
    "dataset",
    "start_date",
    "end_date",
    "output_url",
    "done",
]
continuous_params = [
    "number_of_patient_zero",
    "D_min",
    "D_max",
    "P_max",
    "ITERATIONS",
]
categorical_params = ["infection_model"]
distributions = ["green", "purple", "red", "stable", "intensive_care"]
normal_distributions = [
    "blue",
    "purple_red",
    "purple_pink",
    "pink",
    "stable_white",
    "stable_black",
    "intensive_care_white",
    "intensive_care_black",
]


def make_schema():
    params = {}
    for integer in [
        "D_min",
        "number_of_patient_zero",
        "D_max",
        "infection_model",
    ]:
        params.update({integer: {"type": "integer"}})
    params.update({"P_max": {"type": "number"}})
    params = {"params": {"type": "object", "properties": params}}
    params["params"].update({"required": list(params["params"]["properties"].keys())})

    settings = {
        "settings": {
            "type": "object",
            "properties": {
                k: {"type": "boolean"}
                for k in ["PARALLEL", "LOCAL_TASK", "UPLOAD", "SKIP_TESTS", "VERBOSE"]
            },
        }
    }
    settings["settings"].update(
        {"required": list(settings["settings"]["properties"].keys())}
    )

    meta = {
        "meta": {
            "type": "object",
            "properties": {
                "DATASET": {"type": "string"},
                "ITERATIONS": {"type": "integer"},
                "SENSITIVITY": {"type": "boolean"},
            },
        }
    }
    meta["meta"].update({"required": list(meta["meta"]["properties"].keys())})
    # TODO: add paths
    sensitivity = {
        "sensitivity": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": continuous_params
                        + distributions
                        + normal_distributions,
                    },
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "grouping": {"type": "string"},  # enum
                            "percent": {"type": "integer"},
                            "amount": {"type": "integer"},
                            "max_": {"type": "boolean"},
                            "how": {"type": "string", "enum": ["day", "amount"]},
                        },
                    },
                },
                "ranges": {
                    "type": "object",
                    "properties": {
                        k: {
                            "type": "object",
                            "properties": {
                                kk: {"type": params["params"]["properties"][k]["type"]}
                                for kk in ["step", "min", "max"]
                            },
                        }
                        for k in params["params"]["properties"].keys()
                    },
                },
            },
        }
    }

    schema = {
        "type": "object",
        "properties": {**meta, **settings, **params, **sensitivity},
    }
    return schema


class TestTask(unittest.TestCase):
    def setUp(self):
        self.t = Task(test=True)

    def test_yaml(self):
        for folder in (CONFIG_FOLDER, TEST_FOLDER):
            self.assertTrue((folder / "config.yaml").exists())
            with open(folder / "config.yaml") as f:
                self.yaml_config = load(f, Loader=Loader)
            schema = make_schema()
            # pprint(schema)
            validate(instance=self.yaml_config, schema=schema)

    def test_id(self):
        self.assertLess(self.t.id, 10000000000000000)
        self.assertLessEqual(1000000000000000, self.t.id)
