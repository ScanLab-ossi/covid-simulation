from copy import deepcopy, copy
from pprint import pprint

import streamlit as st
from yaml import load, Loader
import pandas as pd

from simulation.dataset import Dataset
from simulation.task import Task
from simulation.constants import *
from simulation.analysis import Analysis
from simulation.metrics import Metrics
from app.results import get_metrics
from app.helpers import validate_identifier, label_it

with open(CONFIG_FOLDER / "app.yaml") as f:
    config = load(f, Loader=Loader)

with open(CONFIG_FOLDER / "datasets.yaml", "r") as f:
    datasets = load(f, Loader=Loader)

default_task = Task()


def load_inputs(key, new_task):
    element_count = 0
    for n, dict_ in enumerate(config[key].items()):
        param, metadata = dict_
        st.markdown(f"#### {param}")
        element_count += 1
        if metadata["type"] == "categorical":
            options = (
                list(datasets.keys()) if param == "DATASET" else metadata["options"]
            )
            new_task[param] = st.selectbox(
                param, options=options, index=metadata["default_index"],
            )
            element_count += 1
        elif metadata["type"] == "continuous":
            # step = 0.05 if isinstance(v, float) else 1
            new_task[param] = st.number_input(
                param,
                **{
                    "value": default_task[param],
                    "min_value": metadata["range"].get("min"),
                    "max_value": metadata["range"].get("max"),
                    "step": metadata["range"].get("step"),
                },
            )
            element_count += 1
        elif metadata["type"] == "distribution":
            if metadata["dist_type"] == "percent":
                top_placeholder = st.empty()
                # TODO: implement age_dist
            elif metadata["dist_type"] == "normal":
                # step = 0.05 if isinstance(v, float) else 1
                new_task[param] = [
                    st.number_input(
                        "mean", value=default_task[param][0], key=f"{param}_mean",
                    ),
                    st.number_input(
                        "std", value=default_task[param][1], key=f"{param}_std",
                    ),
                ]
                st.write()
                element_count += 3
        if new_task["SENSITIVITY"]:
            if metadata["type"] == "continuous":
                add_to_sa = st.checkbox(
                    "Add this parameter to sensitivity analysis", key=param
                )
                if add_to_sa:
                    if param not in new_task["sensitivity"]["params"]:
                        new_task["sensitivity"]["params"].append(param)
                    for factor in ["min", "max", "step"]:
                        new_task["sensitivity"]["ranges"][param][
                            factor
                        ] = st.number_input(
                            factor,
                            **{
                                "value": default_task["sensitivity"]["ranges"][param][
                                    factor
                                ],
                                "min_value": metadata["range"].get("min"),
                                "max_value": metadata["range"].get("max"),
                                "step": metadata["range"].get("step"),
                                "key": f"{param}_{factor}",
                            },
                        )
                    placeholder = st.empty()
                    validated, message = validate_sensitivity(
                        new_task, error=False, curr_param=param
                    )
                    if not validated:
                        placeholder = st.warning(message)
                    st.markdown("---")
                else:
                    try:
                        new_task["sensitivity"]["params"].remove(param)
                    except ValueError:
                        pass
    return new_task, element_count


def add_sensitivity(new_task):
    new_task["SENSITIVITY"] = st.checkbox("Run sensitivity analysis")
    if new_task["SENSITIVITY"]:
        dataset = Dataset(new_task["DATASET"])
        metrics = Metrics()
        new_task["sensitivity"]["metrics"] = []
        n_metrics = st.number_input(
            "How many metrics do you want to run your sensitivity analysis on?", 1
        )
        for i in range(n_metrics):
            mm, _ = get_metrics(dataset, i)
            new_task["sensitivity"]["metrics"].insert(i, mm)
    return new_task


def sidebar(new_task, extra_params, edit_paths):
    st.sidebar.header("Preview of parameters")
    display_json = {k: v for k, v in new_task.data.items() if k in config["input"]}
    extra_display_json = (
        {k: v for k, v in new_task.data.items() if k in config["extra_input"]}
        if extra_params
        else {}
    )
    st.sidebar.json({"name": new_task["name"], **display_json, **extra_display_json})
    if edit_paths:
        st.sidebar.header("Preview of edited state transition")
        st.sidebar.json(new_task["paths"])
    if new_task["SENSITIVITY"]:
        ss = new_task["sensitivity"]
        active_ranges = {k: v for k, v in ss["ranges"].items() if k in ss["params"]}
        st.sidebar.header("Preview of sensitivity parameters")
        st.sidebar.json(
            {"metrics": ss["metrics"], "params": ss["params"], "ranges": active_ranges}
        )
        times = {
            k: int(((v["max"] - v["min"]) // v["step"]) + 1)
            for k, v in active_ranges.items()
        }
        iters = new_task["ITERATIONS"]
        if len(ss["params"]) > 0:
            st.info(
                f"This sensitivity analysis will run {iters * sum(times.values())} times altogether \
                ({sum(times.values())} times, {iters} iteratons each time). \
                If this seem like a lot, consider making {max(times, key=times.get)}'s range smaller"
            )


# def state_transitions(new_task):
#     p = new_task["paths"]
#     for state, data in p.items():
#         if state in ["white", "black"]:
#             continue
#         st.markdown(f"#### {state} → {', '.join(data['children'])}")
#         if len(data.get("children", [])) == 2:
#             for i in range(2):
#                 p[state]["distribution"][i] = st.number_input(
#                     data["children"][i],
#                     min_value=0.1,
#                     max_value=1.0,
#                     value=data["distribution"][i],
#                     key=f"{state}_distribution_{i}",
#                 )
#         elif len(data.get("children", [])) == 1:
#             for i, what in enumerate(["mean", "std"]):
#                 p[state]["duration"][i] = st.number_input(
#                     f"duration {what}",
#                     value=data["duration"][i],
#                     key=f"{state}_duration_{i}",
#                 )
#     new_task["paths"] = p
#     return new_task


def add_state_transition(task, gcloud):
    options = gcloud.get_filelist()
    st_name = st.selectbox("Select a state transition version", options)
    if options == []:
        st.error("You must upload a state transitions configuration file")
        return task
    else:
        gcloud.download(
            f"state_transitions/{st_name}.csv",
            destination=CONFIG_FOLDER,
            bucket_name="simulation_datasets",
        )
        df = pd.read_csv(CONFIG_FOLDER / f"{st_name}.csv")
        task.load_state_transition(df)
        return task


def load_add_task(gcloud):
    element_count = 0
    new_task = deepcopy(default_task)
    st.title("Add simulation run")
    task_name = st.text_input("Name this simulation run", value=new_task.id)
    task_id_error = st.empty()
    valid_id = validate_identifier(task_name)
    if not valid_id == True:
        task_id_error.error(valid_id)
    else:
        new_task["name"] = task_name
    new_task = add_state_transition(new_task, gcloud)
    # edit = st.radio("", ["Edit task parameters", "Edit state transitions"])
    # reset = st.button("RESET")
    # if reset:
    #     print("reset")
    new_task = add_sensitivity(new_task)
    # st.write("## Edit task parameter")
    _, elem = load_inputs("input", new_task)
    element_count += elem
    # st.write(f"count: {repeat}")
    st.markdown("---")
    extra_params = st.checkbox("Show extra parameters")
    if extra_params:
        # st.write("## Edit extra parameters")
        _, elem = load_inputs("extra_input", new_task)
        element_count += elem
    # st.markdown("---")
    # edit_paths = st.checkbox("Edit state transition")
    # if edit_paths:
    #     new_task = state_transitions(new_task)
    # st.markdown("---")
    placeholder = st.empty()
    sidebar(new_task, extra_params, edit_paths=True)
    submit = st.button("SUBMIT")
    if submit:
        if new_task["SENSITIVITY"]:
            validated, message = validate_sensitivity(new_task)
            if validated:
                try:
                    gcloud.add_tasks([new_task])
                except Exception:
                    pprint(new_task)
                    raise Exception
                placeholder.success(f"Great job! Added new task {label_it(new_task)}")
            else:
                placeholder.error(message)
        else:
            gcloud.add_tasks([new_task])
            placeholder.success(f"Great job! Added new task {label_it(new_task)}")


def is_correct_range(min, max, step):
    return True if int((max * 10 - min * 10)) % int((step * 10)) == 0 else False


def validate_sensitivity(new_task, error=True, curr_param=None):
    params = [curr_param] if curr_param else new_task["sensitivity"]["params"]
    for param in params:
        range_ = new_task["sensitivity"]["ranges"][param]
        if not is_correct_range(**range_):
            message = "Something is wrong in sensitivity range " + (
                f"for {param}" if error else ""
            )
            return False, message
        if not is_correct_range(new_task[param], range_["max"], range_["step"]):
            message = (
                f"Sensitivity range and param {(param if error else '')} do not match"
            )
            return False, message
        if new_task[param] < range_["min"] or new_task[param] > range_["max"]:
            return False, f"Make sure {param} is within sensitivity range"
    return True, ""

