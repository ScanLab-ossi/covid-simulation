from google.cloud import storage, datastore
import pandas as pd
import streamlit as st

from simulation.constants import *

hash_funcs = {
    storage.Client: lambda _: None,
    datastore.Client: lambda _: None,
}


@st.cache(persist=True, hash_funcs=hash_funcs)
def load_data(task_id, gcloud):
    gcloud.download(
        f"{task_id}.csv", destination=OUTPUT_FOLDER, bucket_name="simulation_runs"
    )
    return pd.read_csv(OUTPUT_FOLDER / f"{task_id}.csv")


# @st.cache(hash_funcs=hash_funcs)
# def get_tasklist():
#     gcloud.get_tasklist()
