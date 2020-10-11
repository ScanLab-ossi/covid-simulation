from google.cloud import storage, datastore
import grpc
import pandas as pd
import streamlit as st
from typing import Union

from simulation.constants import *
from simulation.task import Task
from simulation.output import Batch, MultiBatch
from simulation.google_cloud import GoogleCloud

hash_funcs = {
    storage.Client: lambda _: None,
    datastore.Client: lambda _: None,
    grpc._cython.cygrpc.Channel: lambda _: None,
    # bigquery.client.Client: lambda _: None,
}


@st.cache(persist=True, hash_funcs=hash_funcs)
def load_data(task: Task, gcloud: GoogleCloud) -> Union[Batch, MultiBatch]:
    gcloud.download(
        f"{task.id}.csv", destination=OUTPUT_FOLDER, bucket_name="simulation_runs"
    )
    if task["SENSITIVITY"]:
        metabatch = MultiBatch(task)
        metabatch.load()
        return metabatch
    else:
        batch = Batch(task)
        batch.load()
        return batch


def label_it(task: Task) -> str:
    return task.get("name", task.id)


def validate_identifier(task_name):
    if task_name.isidentifier() or task_name.isnumeric():
        return True
    else:
        return "That's not a valid name. Make sure you use \
                underscores instead of spaces, and start with a letter"


# @st.cache(hash_funcs=hash_funcs)
# def get_tasklist():
#     gcloud.get_tasklist()
