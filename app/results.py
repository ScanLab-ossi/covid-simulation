import streamlit as st
from .helpers import load_data

from simulation.visualizer import Visualizer
from simulation.output import Output
from simulation.dataset import Dataset


def results_css():
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        max-width: 100vw;
    }}
    div.element-container > div.fullScreenFrame {{
        text-align: center;
        padding-top: 48px;
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


def load_results(gcloud):
    results_css()
    gcloud.get_tasklist()
    st.title("Simulation results")
    task = st.selectbox(
        "Pick a task id",
        [False] + [t for t in gcloud.done],
        format_func=lambda x: "Please select" if x == False else x.id,
    )
    if task:
        df = load_data(task.id, gcloud).copy()
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
