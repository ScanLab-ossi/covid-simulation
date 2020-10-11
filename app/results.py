import streamlit as st
import numpy as np
import altair as alt

from .helpers import load_data, label_it

from simulation.visualizer import Visualizer
from simulation.output import Batch
from simulation.dataset import Dataset
from simulation.analysis import Analysis
from simulation.metrics import Metrics


def get_metrics(dataset: Dataset, v=None):
    m = Metrics()
    metrics = {}
    metrics["grouping"] = st.selectbox(
        "Select a result to group by",
        options=m.all,
        format_func=Metrics.decrypt_colors,
        key=v,
    )
    percent_or_amount = st.radio(
        "Pick type of threshold to calculate",
        ["specific percent", "specific amount", "max amount", "max percent"],
        key=v,
    )
    if percent_or_amount == "specific percent":
        metrics["max_"] = False
        metrics.pop("amount", None)
        metrics["percent"] = st.number_input(
            "Pick percentage threshold", min_value=0, max_value=100, value=50, key=v
        )
    elif percent_or_amount == "specific amount":
        metrics["max_"] = False
        metrics.pop("percent", None)
        metrics["amount"] = st.number_input(
            f"Pick amount threshold. Note there are {dataset.nodes} nodes in this dataset",
            min_value=0,
            max_value=dataset.nodes,
            value=dataset.nodes // 2,
            key=v,
        )
    else:
        metrics["max_"] = True
        metrics[percent_or_amount.split()[1]] = 1
    metrics["how"] = st.radio(
        "parameter to be returned",
        ["day", "amount"] if metrics["max_"] else ["day"],
        key=v,
    )
    if v != None:
        st.markdown("---")
    return metrics, percent_or_amount


def analysis_count_interface(analysis: Analysis, batch: Batch, dataset: Dataset, task):
    to_analyze, percent_or_amount = get_metrics(dataset)
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
            analysis_count_interface(analysis, batch, dataset, task)
            st.markdown("---")

        st.subheader("Visualize")
        vis = Visualizer(task=task, dataset=dataset, batches=batch)
        if task["SENSITIVITY"]:
            for metric, group in batch.summed.groupby("metric"):
                st.altair_chart(vis._sensitivity_boxplot(group, metric))
        else:
            include_green = st.checkbox("Include green", value=False)
            st.altair_chart(
                vis.visualize(include_green=include_green).properties(
                    width=800, height=500
                )
            )
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
