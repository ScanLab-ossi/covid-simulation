import unittest
import numpy as np
from jsonschema import Draft7Validator, validators, validate

from simulation.task_config import TaskConfig


def make_schema():
    schema = {
        "type": "object",
        "properties": {"age_dist": {"type": "array", "items": {"type": "number"}}},
    }
    for dist in ["blue_to_white", "purple_to_red", "red_to_final_state", "P_r"]:
        schema["properties"].update(
            {
                dist: {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                }
            }
        )
    for integer in ["D_min", "number_of_patient_zero", "D_max"]:
        schema["properties"].update({integer: {"type": "integer"}})
    schema["properties"].update({"P_max": {"type": "number"}})
    schema.update({"required": list(schema["properties"].keys())})
    return schema


test_conf = {
    "age_dist": [0.15, 0.6, 0.25],  # [youngs, adults, olds]
    "blue_to_white": [20, 10],  # ~ Norm(mean, std) | ref:
    "purple_to_red": [5, 2],  # ~ Norm(mean, std)
    "red_to_final_state": [15, 7],
    "D_min": 100,  # Arbitrary, The minimal threshold (in time) for infection,
    "number_of_patient_zero": 10,  # Arbitrary
    "D_max": 700,  # Arbitrary, TO BE CALCULATED,  0.9 precentile of (D_i)'s
    "P_max": 0.05,  # The probability to be infected when the exposure is over the threshold
    "risk_factor": None,  # should be vector of risk by age group
    "P_r": [0.08, 0.03],
}


class TestTaskConfig(unittest.TestCase):
    def setUp(self):
        self.tc = TaskConfig(test_conf)

    def test_params(self):
        validate(instance=self.tc.get_params(), schema=make_schema())

        # with open(Path(DATA_FOLDER / "config_schema.json")) as f:
        #     schema = json.load(f)
        # self.validator = self.create_validator(schema)
        # if validate:
        #     self.validator.validate(self.data)
        #     self.is_valid = self.validator.is_valid(self.data)
        # self.data = self.as_ndarrays()
