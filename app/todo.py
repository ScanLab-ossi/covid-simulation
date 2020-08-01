import streamlit as st
import pandas as pd

from .helpers import label_it


def load_todo(gcloud):
    st.title("Upcoming Tasks")
    gcloud.get_tasklist()

    def render_todo():
        todo = {label_it(x): x for x in gcloud.todo}
        return pd.DataFrame.from_dict(todo, orient="index")  # .to_markdown()

    todo = render_todo()
    if len(todo) > 0:
        # st.table(tasklist)
        st.write(todo)
    else:
        st.success("No tasks left!")

