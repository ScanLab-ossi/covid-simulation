from google.cloud import storage, datastore
import grpc
import pandas as pd
import streamlit as st

from simulation.constants import *
from simulation.task import Task

hash_funcs = {
    storage.Client: lambda _: None,
    datastore.Client: lambda _: None,
    grpc._cython.cygrpc.Channel: lambda _: None,
}


@st.cache(persist=True, hash_funcs=hash_funcs)
def load_data(task_id, gcloud):
    gcloud.download(
        f"{task_id}.csv", destination=OUTPUT_FOLDER, bucket_name="simulation_runs"
    )
    return pd.read_csv(OUTPUT_FOLDER / f"{task_id}.csv")


def label_it(task: Task):
    return task.get("name", task.id)


# @st.cache(hash_funcs=hash_funcs)
# def get_tasklist():
#     gcloud.get_tasklist()
