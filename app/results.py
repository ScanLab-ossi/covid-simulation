import streamlit as st
from .helpers import load_data, label_it
from .style import results_css

from simulation.visualizer import Visualizer
from simulation.output import Output
from simulation.dataset import Dataset


def load_results(gcloud):
    results_css()
    gcloud.get_tasklist()
    st.title("Simulation results")
    task = st.selectbox(
        "Pick a task id",
        [False] + [t for t in gcloud.done],
        format_func=lambda x: "Please select"
        if x == False
        else (f"{label_it(x)} (sensitivity)" if x["SENSITIVITY"] else label_it(x)),
    )
    if task:
        dataset = Dataset(task["DATASET"])
        df = load_data(task.id, gcloud).copy()
        df = df.astype(str) if task["SENSITIVITY"] else df
        vis = Visualizer(output=Output(dataset, task), task=task, dataset=dataset)
        # rename = {"visualize": "graph", "variance_boxplot": "variance"}
        # option = st.radio(
        #     "pick visualiztion",
        #     ["visualize"],  # , "variance_boxplot"],
        #     format_func=lambda x: rename[x],
        # )
        option = "sensitivity_boxplot" if task["SENSITIVITY"] else "visualize"
        # use_container_width=True
        st.altair_chart(
            getattr(vis, option)(df)
        )  # .properties(width=1000, height=600))
        st.sidebar.header("This task's parameters")
        st.sidebar.json(
            {k: v for k, v in task.data.items() if k != "sensitivity"}
            if task["SENSITIVITY"] == False
            else task.data
        )
