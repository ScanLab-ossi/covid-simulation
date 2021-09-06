import unittest
from datetime import datetime

import pandas as pd
import pandas.testing as pdt

from simulation.constants import *
from simulation.dataset import Dataset
from simulation.google_cloud import GoogleCloud
from simulation.infection import GroupInfection, Infection
from simulation.states import States, Variants
from simulation.task import Task

gcloud = GoogleCloud()
task = Task(path=TEST_FOLDER / "config.yaml")
dataset = Dataset(task["DATASET"], task=task, gcloud=gcloud)
infected = [104, 16, 286]
variants = ["variant_a", "variant_b", "variant_a"]
day = 6
final_infected = pd.DataFrame(
    {
        "variant": variants,
        "infection_date": [day] * 3,
        "days_left": [0] * 3,
        "color": ["green"] * 3,
    },
    index=pd.Index(infected, name="infected"),
)


class TestInfection(unittest.TestCase):
    def setUp(self):
        self.infection = Infection(dataset, task)
        self.state_cats = States(task).categories("states")

    def test_organize(self):
        orig = pd.DataFrame(
            {
                "variant": variants,
                "duration": [0.6, 0.3, 0.01],
            },
            index=infected,
        )
        final_infected["color"] = final_infected["color"].astype(self.state_cats)
        pdt.assert_frame_equal(self.infection._organize(orig, day), final_infected)

    def test_r_0(self):
        pass
        # TODO: build test for r_0


class TestGroupInfection(unittest.TestCase):
    def setUp(self):
        self.gi = GroupInfection(dataset, task, reproducible=True)
        self.variant_cats = Variants(task).categories

        self.sample_infected = pd.DataFrame(
            {"duration": [1440, 36, 0], "color": [True, False, False]},
            index=[123] * 3,
        )

    def test_multiply_not_infected_chances(self):
        self.assertAlmostEqual(
            self.gi._multiply_not_infected_chances(self.sample_infected["duration"]),
            0.545,
            places=2,
        )

    def test_infect(self):
        orig = pd.DataFrame(
            {
                "datetime": [datetime.now()] * 2,
                "duration": [200.0, 0.0],
                "group": [{43, 13, 554}, {43, 13, 554}],
                "infector": [43, 43],
                "susceptible": [13, 554],
                "variant": ["variant_a", "variant_a"],
            }
        ).pipe(self.gi.variants.categorify)
        final = (
            pd.DataFrame(
                {
                    "variant": ["variant_a"],
                    "infection_date": [day],
                    "days_left": [0],
                    "color": ["green"],
                },
                index=pd.Index([13], name="infected"),
            )
            .pipe(self.gi.states.categorify)
            .pipe(self.gi.variants.categorify)
        )
        pdt.assert_frame_equal(self.gi._infect(orig, day=day), final)
