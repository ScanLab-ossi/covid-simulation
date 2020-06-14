import unittest
import numpy as np
from jsonschema import Draft7Validator, validators, validate
from yaml import load, Loader
from pprint import pprint

from simulation.task import Task
from simulation.sensitivity_analysis import Analysis
from simulation.constants import *

metadata_keys = [
    "dataset",
    "repetitions",
    "start_date",
    "end_date",
    "output_url",
    "machine_version",
    "done",
]
continuous_params = [
    "number_of_patient_zero",
    "alpha_blue",
    "D_min",
    "D_max",
    "P_max",
]
distribution_params = [
    "age_dist",
    "blue_to_white",
    "purple_to_red",
    "red_to_final_state",
    "P_r",
]
categorical_params = ["infection_model"]


def make_schema():
    params = {"age_dist": {"type": "array", "items": {"type": "number"}}}
    for dist in ["blue_to_white", "purple_to_red", "red_to_final_state", "P_r"]:
        params.update(
            {
                dist: {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
            }
        )
    for integer in [
        "D_min",
        "number_of_patient_zero",
        "D_max",
        "infection_model",
    ]:
        params.update({integer: {"type": "integer"}})
    params.update({k: {"type": "number"} for k in ["P_max", "alpha_blue"]})
    params = {"params": {"type": "object", "properties": params}}
    params["params"].update({"required": list(params["params"]["properties"].keys())})

    settings = {
        "settings": {
            "type": "object",
            "properties": {
                k: {"type": "boolean"}
                for k in ["PARALLEL", "LOCAL", "UPLOAD", "SKIP_TESTS", "VERBOSE"]
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

    sensitivity = {
        "sensitivity": {
            "type": "object",
            "properties": {
                "params": {
                    "type": "array",
                    "items": {"type": "string", "enum": continuous_params},
                },
                "metric": {
                    "type": "string",
                    "enum": [attr for attr in dir(Analysis) if attr[:2] != "__"],
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
        self.t = Task()

    def test_yaml(self):
        self.assertTrue((CONFIG_FOLDER / "config.yaml").exists())
        with open(CONFIG_FOLDER / "config.yaml") as f:
            self.yaml_config = load(f, Loader=Loader)
        schema = make_schema()
        # pprint(schema)
        validate(instance=self.yaml_config, schema=schema)

    def test_id(self):
        self.assertLess(self.t.id, 10000000000000000)
        self.assertLessEqual(1000000000000000, self.t.id)

    def test_get_machine_version(self):
        self.assertTrue(self.t.get_machine_version().isalnum())
        self.assertTrue(len(self.t.get_machine_version()), 7)
