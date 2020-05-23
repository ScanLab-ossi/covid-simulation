import json
from datetime import datetime, date, timedelta
import numpy as np
from pathlib import Path

from simulation.helpers import timing, one_array_pickle_to_set
from simulation.constants import *
from simulation.basic_configuration import BasicConfiguration
from simulation.task_config import TaskConfig
from simulation.google_cloud import GoogleCloud
from simulation.dataset import Dataset
from simulation.state_transition import StateTransition
from simulation.contagion import CSVContagion, SQLContagion
from simulation.output import Output
from simulation.visualizer import Visualizer

# TODO: change np.arrays to lists

test_conf = {
    "age_dist": np.array([0.15, 0.6, 0.25]),  # [youngs, adults, olds]
    "recovery_time_dist": np.array(
        [20, 10]
    ),  # recovery_time_dist ~ Norm(mean, std) | ref:
    "aggravation_time_dist": np.array(
        [5, 2]
    ),  # aggravation_time_dist ~ Norm(mean, std) | ref:
    "D_min": 10,  # Arbitrary, The minimal threshold (in time) for infection,
    "number_of_patient_zero": 10,  # Arbitrary
    "D_max": 70,  # Arbitrary, TO BE CALCULATED,  0.9 precentile of (D_i)'s
    "P_max": 0.2,  # The probability to be infected when the exposure is over the threshold
    "risk_factor": None,  # should be vector of risk by age group
}


class ContagionRunner(object):
    @staticmethod
    @timing
    def contagion_runner(dataset, output, basic_conf, contagion, task_conf=test_conf):
        st = StateTransition(dataset, task_conf)
        if dataset.storage == "sql":
            pickle_of_ids = one_array_pickle_to_set(
                Path(DATA_FOLDER / "destination_ids_first_3days.pickle")
            )
            zero_patients = contagion.pick_patient_zero(
                pickle_of_ids, num_of_patients=task_conf.get("number_of_patient_zero")
            )
        else:
            zero_patients = contagion.pick_patient_zero()
        st.get_trajectory(
            zero_patients, output, dataset.start_date, add_duration=dataset.add_duration
        )
        set_of_infected = zero_patients
        for day in range(dataset.period + 1):
            print(f"Status of {day}:")
            curr_date = dataset.start_date + timedelta(days=day)
            # TODO: make contagion in sql and contagion in csv have same input
            set_of_contact_with_patient = contagion.contagion(
                set_of_infected, curr_date=curr_date
            )
            st.get_trajectory(
                set_of_contact_with_patient,
                output,
                curr_date,
                add_duration=dataset.add_duration,
            )
            set_of_infected = set(
                output.df[output.df["expiration_date"] > curr_date].index
            )
            # patients that haven't recovered or died yet
            output.shape()

        output.export()


def main(test_conf: dict = False, test=True):
    basic_conf = BasicConfiguration()
    gcloud = GoogleCloud(basic_conf)
    dataset = Dataset(DATASET)
    dataset.load_dataset(gcloud=gcloud)
    if test_conf:
        output = Output(dataset=dataset)
        contagion = (
            CSVContagion(dataset, test_conf)
            if dataset.storage == "csv"
            else SQLContagion(dataset, test_conf, basic_conf)
        )
        ContagionRunner.contagion_runner(
            dataset=dataset,
            task_conf=test_conf,
            output=output,
            basic_conf=basic_conf,
            contagion=contagion,
        )
        visualizer = Visualizer(output)
        visualizer.visualize()
        if not test:
            task_key = gcloud.add_task(dataset.name, dict(TaskConfig), done=True)
            gcloud.upload(output.output_path, new_name=task_key)
    else:
        gcloud.get_tasklist()
        if gcloud.todo:
            results = []
            for task in gcloud.todo:
                data = Output(output_path=task.id)
                ContagionRunner.contagion_runner(
                    data, task_conf=TaskConfig(task), basic_conf=basic_conf
                )
                results.append((data, task))
            gcloud.write_results(results)
        else:
            print("No tasks waiting!")


if __name__ == "__main__":
    main(test_conf)
