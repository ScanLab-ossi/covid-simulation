import streamlit as st
import numpy as np
import pandas as pd
import json
from copy import copy
from datetime import date
from google.cloud import storage, datastore

from simulation.google_cloud import GoogleCloud
from simulation.basic_configuration import BasicConfiguration
from simulation.task import Task
from simulation.constants import *
from simulation.dataset import Dataset
from simulation.visualizer import Visualizer
from simulation.output import Output

with open(CONFIG_FOLDER / "app.yaml") as f:
    config = load(f, Loader=Loader)

gcloud = GoogleCloud(BasicConfiguration())
gcloud.get_tasklist()
default_task = Task()
datasets = Dataset("mock_data").datasets


# TODO:
# configurable age_dist
# state+default as dict
# config validation

hash_funcs = {
    storage.Client: lambda x: x.project + "_storage",
    datastore.Client: lambda x: x.project + "_datastore",
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
    get_tasklist()
    st.header("Show Simulation Results")
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
        st.altair_chart(getattr(vis, option)(df), use_container_width=True)
        st.sidebar.header("This task's parameters")
        st.sidebar.json(task.data)
    # st.altair_chart()

    # st.image(OUTPUT_FOLDER / f"{task_id}.csv", use_column_width=True)


def load_todo():
    st.header("Upcoming Tasks")
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


def load_add_task():
    new_task = copy(default_task)
    st.header("Add simulation run")
    for type_, params in config["input"].items():
        # if k not in ("machine_version", "start_date", "done"):
        if isinstance(params, dict):
            for param, default in params.items():
                # st.sidebar.markdown(f"#### {param}")
                if type_ == "categorical":
                    # st.sidebar.markdown(f"{v}")
                    options = list(datasets.keys()) if param == "DATASET" else [1, 2]
                    new_task[param] = st.selectbox(
                        param, options=options, index=default
                    )
                elif type_ == "continuous":
                    # step = 0.05 if isinstance(v, float) else 1
                    new_task[param] = st.number_input(
                        param,
                        **{
                            "value": default_task[param],
                            "min_value": default.get("min"),
                            "max_value": default.get("max"),
                            "step": default.get("step"),
                            "key": f"{param}",
                        },
                    )
                elif type_ == "distribution":
                    if default == "percent":
                        top_placeholder = st.empty()
                        #             if state[1][1] + state[2][1] + state[3][1] != 1.0:
                        # top_placeholder.warning("should add up to 100!")

                    elif default == "normal":
                        st.markdown(f"#### {param}")
                        # step = 0.05 if isinstance(v, float) else 1
                        new_task[param] = [
                            st.number_input(
                                "mean",
                                value=default_task[param][0],
                                key=f"{param}_mean",
                            ),
                            st.number_input(
                                "std", value=default_task[param][1], key=f"{param}_std"
                            ),
                        ]
        else:
            new_task["SENSITIVITY"] = st.checkbox("Run sensitivity analysis?")
            if new_task["SENSITIVITY"]:
                st.json(default_task["sensitivity"])
    st.sidebar.header("Preview")
    st.sidebar.json(new_task.data)
    submit = st.button("submit")
    placeholder = st.empty()

    # st.json({k: v for k, v in task.items() if not isinstance(v, date)})

    if submit:
        if validate("a"):
            new_task_id = gcloud.add_tasks([new_task])
            placeholder.success(f"great job! added new task {new_task_id[0]}")
            # send to datastore
        else:
            placeholder.error("doesn't match up. try again!")


def validate(state):
    return True


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
