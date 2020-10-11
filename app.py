import streamlit as st
import pandas as pd
from datetime import date
import altair as alt

from simulation.google_cloud import GoogleCloud
from simulation.constants import *

from app.style import streamlit_theme, nav_css, grid_css
from app.results import load_results
from app.todo import load_todo
from app.add_task import load_add_task
from app.helpers import load_data
from app.edit_state_transition import load_edit_state_transition

alt.themes.register("streamlit", streamlit_theme)
alt.themes.enable("streamlit")

gcloud = GoogleCloud()
gcloud.get_tasklist()

# TODO:
# fix name!
# more config validation?
# dataset metadata
# sensitivity results
# testing?
# crud for tasks?
# tasklist display to show full json per task?


nav_css()
nav = st.radio(
    "navigation", ["Add Task", "Edit State Transition", "Results", "Tasklist"], index=0
)
if nav == "Results":
    load_results(gcloud)
elif nav == "Tasklist":
    load_todo(gcloud)
elif nav == "Add Task":
    load_add_task(gcloud)
elif nav == "Edit State Transition":
    load_edit_state_transition(gcloud)


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
