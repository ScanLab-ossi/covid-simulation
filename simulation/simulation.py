import json, os, sys
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from typing import Union

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
    def contagion_runner(
        dataset: Dataset,
        output: Output,
        basic_conf: BasicConfiguration,
        contagion: Union[CSVContagion, SQLContagion],
        task_conf: TaskConfig = test_conf,
        repeat: int = 1,
    ):
        for i in range(repeat):
            st = StateTransition(dataset, task_conf)
            if dataset.storage == "sql":
                pickle_of_ids = one_array_pickle_to_set(
                    Path(DATA_FOLDER / "destination_ids_first_3days.pickle")
                )
                zero_patients = contagion.pick_patient_zero(
                    pickle_of_ids,
                    num_of_patients=task_conf.get("number_of_patient_zero"),
                )
            else:
                zero_patients = contagion.pick_patient_zero()
            st.get_trajectory(
                zero_patients,
                output,
                dataset.start_date,
                add_duration=dataset.add_duration,
            )
            set_of_infected = zero_patients
            for day in range(dataset.period + 1):
                process = f", process {os.getpid()}" if PARALLEL else ""
                print(f"Status of {day}:" + process)
                curr_date = dataset.start_date + timedelta(days=day)
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
            output.summed.append(output.sum_output())
            output.reset()
        output.average_outputs()
        output.export()
        return output.average


def main(test_conf: dict = False):
    print(
        f"""starting!
DATASET = {DATASET}
REPETITIONS = {REPETITIONS}
UPLOAD = {UPLOAD}
PARALLEL = {PARALLEL}
LOCAL = {LOCAL}
"""
    )
    basic_conf = BasicConfiguration()
    gcloud = GoogleCloud(basic_conf)
    if not LOCAL:
        gcloud.get_tasklist()
        if not gcloud.todo:
            print("you've picked LOCAL=False, but no tasks are waiting")
            sys.exit()
    results = []
    tasklist = [test_conf] if LOCAL else gcloud.todo
    for task in tasklist:
        dataset = Dataset(DATASET) if LOCAL else task.dataset
        dataset.load_dataset(gcloud=gcloud)
        output = (
            Output(dataset=dataset)
            if LOCAL
            else Output(dataset=dataset, output_filename=task.id)
        )
        contagion = (
            CSVContagion(dataset, test_conf)
            if dataset.storage == "csv"
            else SQLContagion(dataset, test_conf, basic_conf)
        )
        if PARALLEL:
            with mp.Pool() as p:
                r = [
                    p.apply_async(
                        ContagionRunner.contagion_runner,
                        (dataset, output, basic_conf, contagion, task),
                    )
                    for _ in range(REPETITIONS)
                ]
                output.summed = [res.get() for res in r]
                output.average_outputs()
                output.export()
        else:
            ContagionRunner.contagion_runner(
                dataset, output, basic_conf, contagion, task, repeat=REPETITIONS
            )
        visualizer = Visualizer(output)
        visualizer.visualize()
        if UPLOAD:
            if len(tasklist) > 1:
                results.append((output, task))
                continue
            task_key = gcloud.add_task(dataset.name, TaskConfig(test_conf), done=True)
            gcloud.upload(output.output_path, new_name=task_key)
    if len(tasklist) > 1 and UPLOAD:
        gcloud.write_results(results)


if __name__ == "__main__":
    main(test_conf)
