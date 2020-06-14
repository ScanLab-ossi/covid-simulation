import numpy as np
from datetime import timedelta, date
import pandas as pd
from typing import Tuple, Union

from simulation.dataset import Dataset
from simulation.task import Task
from simulation.output import Output
from simulation.helpers import timing


class StateTransition(object):
    def __init__(self, dataset: Dataset, task: Task):
        self.task = task
        self.dataset = dataset

    def _check_if_aggravate(self, age_group: np.ndarray) -> np.ndarray:
        # TO BE MORE COMPETABILE TO THE MODEL
        return np.random.rand(len(age_group)) > self.task["S_i"]

    def _get_next_date(self, dist_and_previous_date: Tuple[str, date]) -> date:
        dist, previous_date = dist_and_previous_date
        if dist == "same":
            return previous_date
        duration = int(np.maximum(np.around(np.random.normal(*self.task[dist])), 1))
        next_date = previous_date + timedelta(duration)
        if next_date > self.dataset.end_date:
            return self.dataset.end_date
        return next_date

    def _final_state(self, color: bool) -> str:
        if color:
            P_r = np.random.normal(*self.task["P_r"])
            return np.random.choice(["w", "k"], 1, p=[1 - abs(P_r), abs(P_r)]).item()
        else:
            return "w"

    def _infection_state_transition(
        self, infected: pd.DataFrame, infection_date: date
    ) -> pd.DataFrame:
        # get info about an infected person and return relevant data for dataframe
        df = pd.DataFrame(
            np.random.choice(
                len(self.task.get("age_dist")),
                len(infected.index),
                p=self.task.get("age_dist"),
            ),
            columns=["age_group"],
            index=infected.index,
        )
        # True=purple, False=blue
        df["color"] = self._check_if_aggravate(df["age_group"].values)
        df["infection_date"] = infection_date
        df["transition_date"] = (
            df[["color", "infection_date"]]
            .replace({True: "blue_to_white", False: "purple_to_red"})
            .apply(self._get_next_date, axis=1)
        )
        df["expiration_date"] = (
            df[["color", "transition_date"]]
            .replace({True: "red_to_final_state", False: "same"})
            .apply(self._get_next_date, axis=1)
        )
        df["final_state"] = df["color"].apply(self._final_state)
        return df

    def get_trajectory(
        self, infected: Union[pd.DataFrame, set], output: Output, curr_date: date
    ) -> pd.DataFrame:
        if isinstance(infected, set):
            infected = pd.DataFrame(index=infected)
        # remove newly infected if they've already been infected in the past
        infected = infected.loc[~infected.index.isin(output.df.index)]
        # no need to add anything if there are no newly infected today
        if len(infected) == 0:
            return infected
        infected = infected.join(self._infection_state_transition(infected, curr_date))
        return infected
