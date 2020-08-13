import streamlit as st
from copy import deepcopy, copy
import json
from yaml import load, Loader
from pprint import pprint

from simulation.dataset import Dataset
from simulation.task import Task
from simulation.constants import *
from simulation.sensitivity_analysis import Analysis
from app.session_state import get as get_session

with open(CONFIG_FOLDER / "app.yaml") as f:
    config = load(f, Loader=Loader)

with open(CONFIG_FOLDER / "datasets.json", "r") as f:
    datasets = json.load(f)

default_task = Task()
pprint(default_task.data)
session = get_session(run_id=0)


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
                param,
                options=options,
                index=metadata["default_index"],
                key=session.run_id,
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
                    "key": session.run_id,
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
                        "mean",
                        value=default_task[param][0],
                        key=f"{param}_mean_{session.run_id}",
                    ),
                    st.number_input(
                        "std",
                        value=default_task[param][1],
                        key=f"{param}_std_{session.run_id}",
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
                                "key": f"{param}_{factor}_{session.run_id}",
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


def load_add_task(gcloud):
    element_count = 0
    new_task = deepcopy(default_task)
    st.title("Add simulation run")
    task_name = st.text_input("Name this simulation run", value=new_task.id)
    task_id_error = st.empty()
    if not validate_task_id(task_name):
        task_id_error.error(
            "That's not a valid task name. Make sure you use \
            underscores instead of spaces, and start with a letter"
        )
    else:
        new_task["name"] = task_name
    reset = st.button("RESET")
    if reset:
        session.run_id += 1
    new_task["SENSITIVITY"] = st.checkbox("Run sensitivity analysis")
    if new_task["SENSITIVITY"]:
        new_task["sensitivity"]["metric"] = st.selectbox(
            "Pick metric to run sensitivity analysis on:",
            [x for x in dir(Analysis) if x[0] != "_"],
            index=1,
        )
        st.markdown("---")
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
    # grid_css(element_count)
    st.sidebar.header("Preview of parameters")

    display_json = {k: v for k, v in new_task.data.items() if k in config["input"]}
    extra_display_json = (
        {k: v for k, v in new_task.data.items() if k in config["extra_input"]}
        if extra_params
        else {}
    )
    st.sidebar.json({"name": new_task["name"], **display_json, **extra_display_json})
    if new_task["SENSITIVITY"]:
        ss = new_task["sensitivity"]
        active_ranges = {k: v for k, v in ss["ranges"].items() if k in ss["params"]}
        st.sidebar.header("Preview of sensitivity parameters")
        st.sidebar.json(
            {"metric": ss["metric"], "params": ss["params"], "ranges": active_ranges}
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
    # st.markdown("---")
    placeholder = st.empty()
    submit = st.button("SUBMIT")
    if submit:
        if new_task["SENSITIVITY"]:
            validated, message = validate_sensitivity(new_task)
            if validated:
                gcloud.add_tasks([new_task])
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


def validate_task_id(task_name):
    return True if task_name.isidentifier() or task_name.isnumeric() else False

