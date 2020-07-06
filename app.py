import streamlit as st
import numpy as np
import pandas as pd
import json
from copy import copy
from datetime import date
from google.cloud import storage, datastore
import altair as alt

from simulation.google_cloud import GoogleCloud
from simulation.basic_configuration import BasicConfiguration
from simulation.task import Task
from simulation.constants import *
from simulation.dataset import Dataset
from simulation.visualizer import Visualizer
from simulation.output import Output
from simulation.sensitivity_analysis import Analysis

from app.extras import streamlit_theme, results_css


alt.themes.register("streamlit", streamlit_theme)
alt.themes.enable("streamlit")

with open(CONFIG_FOLDER / "app.yaml") as f:
    config = load(f, Loader=Loader)

bc = BasicConfiguration()
gcloud = GoogleCloud()
gcloud.get_tasklist()
default_task = Task()
datasets = Dataset("mock_data").datasets


# root > div:nth-child(1) > div > div > div > div > section.main > div > div:nth-child(1) > div:nth-child(7) > div {{


# TODO:
# more config validation?
# dataset metadata
# sensitivity results
# testing?
# crud for tasks?
# tasklist display to show full json per task?

hash_funcs = {
    storage.Client: lambda _: None,
    datastore.Client: lambda _: None,
}


@st.cache(persist=True)
def load_data(task_id):
    gcloud.download(
        f"{task_id}.csv", destination=OUTPUT_FOLDER, bucket_name="simulation_runs"
    )
    return pd.read_csv(OUTPUT_FOLDER / f"{task_id}.csv")


@st.cache(hash_funcs=hash_funcs)
def get_tasklist():
    gcloud.get_tasklist()


def load_results():
    results_css()
    get_tasklist()
    st.title("Simulation results")
    task = st.selectbox(
        "Pick a task id",
        [False] + [t for t in gcloud.done],
        format_func=lambda x: "Please select" if x == False else x.id,
    )
    if task:
        df = load_data(task.id).copy()
        vis = Visualizer(output=Output(Dataset(task["DATASET"]), task=task), task=task)
        # rename = {"visualize": "graph", "variance_boxplot": "variance"}
        # option = st.radio(
        #     "pick visualiztion",
        #     ["visualize"],  # , "variance_boxplot"],
        #     format_func=lambda x: rename[x],
        # )
        option = "visualize"
        st.altair_chart(
            getattr(vis, option)(df).properties(
                width=1000, height=600
            )  # use_container_width=True
        )
        st.sidebar.header("This task's parameters")
        st.sidebar.json(
            {k: v for k, v in task.data.items() if k != "sensitivity"}
            if task["SENSITIVITY"] == False
            else task.data
        )


def load_todo():
    st.title("Upcoming Tasks")
    get_tasklist()

    def render_todo():
        todo = {x.id: x for x in gcloud.todo}
        return pd.DataFrame.from_dict(todo, orient="index")  # .to_markdown()

    todo = render_todo()
    if len(todo) > 0:
        # st.table(tasklist)
        st.write(todo)
    else:
        st.success("No tasks left!")


def load_inputs(key, new_task):
    for n, dict_ in enumerate(config[key].items()):
        param, metadata = dict_
        st.markdown(f"#### {param}")
        if metadata["type"] == "categorical":
            options = (
                list(datasets.keys()) if param == "DATASET" else metadata["options"]
            )
            new_task[param] = st.selectbox(
                param, options=options, index=metadata["default_index"]
            )
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
                        "std", value=default_task[param][1], key=f"{param}_std"
                    ),
                ]
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
                    validated, message = validate(new_task, error=False)
                    if not validated:
                        placeholder = st.warning(message)
                    st.markdown("---")
                else:
                    try:
                        new_task["sensitivity"]["params"].remove(param)
                    except ValueError:
                        pass
    return new_task


def load_add_task():
    new_task = copy(default_task)
    st.title("Add simulation run")
    new_task["SENSITIVITY"] = st.checkbox("Run sensitivity analysis")
    if new_task["SENSITIVITY"]:
        new_task["sensitivity"]["metric"] = st.selectbox(
            "Pick metric to run sensitivity analysis on:",
            [x for x in dir(Analysis) if x[0] != "_"],
            index=1,
        )
        st.markdown("---")
    # st.write("## Edit task parameter")
    load_inputs("input", new_task)
    # st.markdown("---")
    extra_params = st.checkbox("Show extra parameters")
    if extra_params:
        # st.write("## Edit extra parameters")
        load_inputs("extra_input", new_task)
    st.sidebar.header("Preview of parameters")

    display_json = {k: v for k, v in new_task.data.items() if k in config["input"]}
    extra_display_json = (
        {k: v for k, v in new_task.data.items() if k in config["extra_input"]}
        if extra_params
        else {}
    )
    st.sidebar.json({**display_json, **extra_display_json})
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
    st.markdown("---")
    submit = st.button("SUBMIT")
    placeholder = st.empty()
    if submit:
        validated, message = validate(new_task)
        if validated:
            new_task_id = gcloud.add_tasks([new_task])
            placeholder.success(f"Great job! Added new task {new_task_id[0]}")
        else:
            placeholder.error(message)


def is_correct_range(min, max, step):
    return True if int((max * 10 - min * 10)) % int((step * 10)) == 0 else False


def validate(new_task, error=True):
    for param, range_ in new_task["sensitivity"]["ranges"].items():
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


nav = st.selectbox("navigation", ["Add Task", "Results", "Tasklist"], index=0)
if nav == "Results":
    load_results()
elif nav == "Tasklist":
    load_todo()
elif nav == "Add Task":
    load_add_task()


# fig, ax = plt.subplots()

# s = np.random.normal(mu, sigma)
# count, bins, ignored = plt.hist(s, 30, density=True)

# (line,) = ax.plot(
#     bins,
#     1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((bins - mu) ** 2) / (2 * sigma ** 2)),
#     linewidth=2,
#     color="r",
# )

# the_plot = st.pyplot(plt)
