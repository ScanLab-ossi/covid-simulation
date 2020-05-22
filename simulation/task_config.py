from pathlib import Path
import json
from jsonschema import Draft7Validator, validators
import numpy as np
from collections import UserDict

from simulation.constants import *


class TaskConfig(UserDict):
    """
    Config for each simulation run. Can be used for validation
    """

    def __init__(self, data, validate=True):
        super().__init__(dict(data))
        with open(Path(DATA_FOLDER / "config_schema.json")) as f:
            schema = json.load(f)
        self.validator = self.create_validator(schema)
        if validate:
            self.validator.validate(self.data)
            self.is_valid = self.validator.is_valid(self.data)
        self.data = self.as_ndarrays()

    def create_validator(self, schema):
        type_checker = Draft7Validator.TYPE_CHECKER.redefine(
            "array",
            fn=lambda checker, instance: True
            if isinstance(instance, np.ndarray) or isinstance(instance, list)
            else False,
        )
        Validator = validators.extend(Draft7Validator, type_checker=type_checker)
        return Validator(schema=schema)

    def as_lists(self):
        return {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in self.data.items()
        }

    def as_ndarrays(self):
        return {
            k: (np.array(v) if isinstance(v, list) else v)
            for k, v in dict(self.data).items()
        }
