import streamlit as st
import numpy as np
import altair as alt

from .helpers import load_data, label_it
from .style import results_css

from simulation.visualizer import Visualizer
from simulation.output import Batch
from simulation.dataset import Dataset
from simulation.analysis import Analysis
from simulation.constants import color_dict


def analysis_count_interface(analysis, batch, dataset):
    to_analyze = {}
    to_analyze["grouping"] = st.selectbox(
        "Select a result to group by (color, sick, infected, or infectors)",
        options=analysis.not_colors + list(color_dict),
    )
    percent_or_amount = st.radio(
        "Pick type of threshold to calculate",
        ["specific percent", "specific amount", "max amount", "max percent"],
    )
    if percent_or_amount == "specific percent":
        to_analyze["max_"] = False
        to_analyze.pop("amount", None)
        to_analyze["percent"] = st.number_input(
            "Pick percentage threshold", min_value=0, max_value=100, value=50
        )
    elif percent_or_amount == "specific amount":
        to_analyze["max_"] = False
        to_analyze.pop("percent", None)
        to_analyze["amount"] = st.number_input(
            f"Pick amount threshold. Note there are {dataset.nodes} nodes in this dataset",
            min_value=0,
            max_value=dataset.nodes,
            value=dataset.nodes // 2,
        )
    else:
        to_analyze["max_"] = True
        to_analyze[percent_or_amount.split()[1]] = 1
    to_analyze["how"] = st.radio(
        "parameter to be returned",
        ["day", "amount"] if to_analyze["max_"] else ["day"],
    )
    res = analysis.count(batch, **to_analyze)
    if res != np.inf:
        percent_or_amount = percent_or_amount.split()[1]
        if to_analyze["max_"]:
            st.success(
                f"""On average, the {percent_or_amount} of {to_analyze['grouping']} reached it's maximum
                    {f'on day number **{res:.2f}**' if to_analyze['how'] == 'day' else f'with the amount of **{res:.2f}**'}"""
            )
        else:
            st.success(
                f"On average, **{res:.2f}** is the {to_analyze['how']} where the {percent_or_amount} \
                of {to_analyze['grouping']} surpassed \
                {str(to_analyze['percent']) + '%' if percent_or_amount == 'percent' else to_analyze['amount']}"
            )
    else:
        st.error("nothing satisfies these constraints")


def load_results(gcloud):
    # results_css()
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
        batch = load_data(task, gcloud)
        # df = df if task["SENSITIVITY"] else df

        st.markdown("---")
        if not task["SENSITIVITY"]:
            batch.average_outputs()
            analysis = Analysis(dataset=dataset, task=task)
            st.subheader("Show specific results")
            analysis_count_interface(analysis, batch, dataset)
            st.markdown("---")

        st.subheader("Visualize")
        vis = Visualizer(task=task, dataset=dataset, batches=batch, save=True)
        if task["SENSITIVITY"]:
            for metric, group in batch.summed.groupby("metric"):
                st.altair_chart(vis._sensitivity_boxplot(group, metric))
        else:
            st.altair_chart(vis.visualize().properties(height=500))
        # st.altair_chart(
        #     alt.vconcat(
        #         *[
        #             vis._sensitivity_boxplot(group, metric)
        #             for metric, group in batch.summed.groupby("metric")
        #         ]
        #     )
        # )
        st.sidebar.header("This task's parameters")
        st.sidebar.json(
            {k: v for k, v in task.data.items() if k != "sensitivity"}
            if task["SENSITIVITY"] == False
            else task.data
        )
