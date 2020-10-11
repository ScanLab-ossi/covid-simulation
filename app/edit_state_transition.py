from io import TextIOWrapper
import streamlit as st
import pandas as pd

from simulation.task import Task
from simulation.constants import *
from simulation.google_cloud import GoogleCloud
from app.helpers import validate_identifier

st.set_option("deprecation.showfileUploaderEncoding", False)


def load_edit_state_transition(gcloud: GoogleCloud):
    new_task = Task()
    st.markdown("### Add new state transition config")
    file_buffer = st.file_uploader("Upload new state transition", type="csv")
    st_name = st.text_input("Name this state transition")
    st_name_error = st.empty()
    valid_st_name = validate_identifier(st_name)
    if not valid_st_name == True:
        st_name_error.error(valid_st_name)
    if file_buffer is not None and valid_st_name == True:
        text_buffer = TextIOWrapper(file_buffer)
        df = pd.read_csv(file_buffer)
        st.write(df.set_index("state"))
        df.to_csv(CONFIG_FOLDER / f"{st_name}.csv", index=False)
        upload = st.button("UPLOAD")
        if upload:
            gcloud.upload(
                CONFIG_FOLDER / f"{st_name}.csv",
                new_name=f"state_transitions/{st_name}",
                bucket_name="simulation_datasets",
            )
            st.success(
                f"Great job! Uploaded state transition configuration **{st_name}**."
            )
    st.markdown("---")
    st.markdown("### Display a state transition config")
    options = gcloud.get_filelist()
    st_name = st.selectbox("Select a state transition version", options)
    if options == []:
        st.error("You must upload a state transitions configuration file")
        return task
    else:
        gcloud.download(
            f"state_transitions/{st_name}.csv",
            destination=CONFIG_FOLDER,
            bucket_name="simulation_datasets",
        )
        st.write(pd.read_csv(CONFIG_FOLDER / f"{st_name}.csv").set_index("state"))

