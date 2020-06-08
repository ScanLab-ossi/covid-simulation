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

test_conf = {
    "age_dist": [0.15, 0.6, 0.25],  # [youngs, adults, olds]
    "blue_to_white": [20, 10],  # ~ Norm(mean, std) | ref:
    "purple_to_red": [5, 2],  # ~ Norm(mean, std)
    "red_to_final_state": [15, 7],
    "number_of_patient_zero": 10,  # Arbitrary
    "alpha_blue": 0.5,  # if alpha_blue == 1 it will be skipped
    "D_min": 2,  # Arbitrary, The minimal threshold (in time) for infection,
    "D_max": 1440,  # Arbitrary, TO BE CALCULATED,  0.9 precentile of (D_i)'s
    "P_max": 0.8,  # The probability to be infected when the exposure is over the threshold
    "threshold": 0.05,
    "infection_model": 2,
    "S_i": 0.7,  # should be vector of risk by age group
    "P_r": [0.08, 0.03],
}


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
        start = datetime.now()
        if not PARALLEL:
            print(f"repetition {i}")
        st = StateTransition(dataset, task_conf)
        if dataset.storage == "sql":
            pickle_of_ids = one_array_pickle_to_set(
                Path(DATA_FOLDER / "destination_ids_first_3days.pickle")
            )
            zero_patients = contagion.pick_patient_zero(
                pickle_of_ids, num_of_patients=task_conf.get("number_of_patient_zero"),
            )
        else:
            zero_patients = contagion.pick_patient_zero()
        st.get_trajectory(
            zero_patients, output, dataset.start_date,
        )
        infected_df = output.df[["color"]]
        for day in range(dataset.period + 1):
            if VERBOSE:
                process = f", process {os.getpid()}" if PARALLEL else ""
                print(f"Status of {day}:" + process)
            curr_date = dataset.start_date + timedelta(days=day)
            set_of_contact_with_patient = contagion.contagion(
                infected_df, curr_date=curr_date
            )
            st.get_trajectory(set_of_contact_with_patient, output, curr_date)
            infected_df = output.df[output.df["transition_date"] > curr_date][["color"]]
            # patients that haven't recovered or died yet
            if VERBOSE:
                print(f"{output.df.shape[0]} infected today altogether")
        output.summed.append(output.sum_output())
        output.reset()
        print(f"repetition {i} took {datetime.now()- start}")
    output.concat_outputs()
    # output.export(how="concated")
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
VERBOSE = {VERBOSE}"""
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
                        contagion_runner,
                        (dataset, output, basic_conf, contagion, task),
                    )
                    for _ in range(REPETITIONS)
                ]
                output.summed = [res.get() for res in r]
                output.average_outputs()
                output.export()
        else:
            contagion_runner(
                dataset, output, basic_conf, contagion, task, repeat=REPETITIONS
            )
        visualizer = Visualizer(output)
        visualizer.visualize()
        if REPETITIONS > 1:
            visualizer.boxplot_variance()
        if UPLOAD:
            if len(tasklist) > 1:
                results.append((output, task))
                continue
            task_key = gcloud.add_task(dict(TaskConfig(test_conf)), done=True)
            gcloud.upload(output.csv_path, new_name=task_key)
    if len(tasklist) > 1 and UPLOAD:
        gcloud.write_results(results)


if __name__ == "__main__":
    main(test_conf)
