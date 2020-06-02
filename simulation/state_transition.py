import numpy as np
from datetime import timedelta
import pandas as pd

from simulation.dataset import Dataset
from simulation.task_config import TaskConfig
from simulation.helpers import timing


class StateTransition(object):
    def __init__(self, dataset: Dataset, task_conf: TaskConfig):
        self.task_conf = task_conf
        self.dataset = dataset

    def _get_next_date(self, dist_and_previous_date):
        dist, previous_date = dist_and_previous_date
        if dist == "same":
            return previous_date
        duration = int(np.around(np.random.normal(*self.task_conf.get(dist))))
        if duration <= 1:  # Avoid the paradox of negative recovery duration.
            duration = 1
        next_date = previous_date + timedelta(duration)
        if next_date > self.dataset.end_date:
            return self.dataset.end_date
        return next_date

    def _check_if_aggravate(self, age_group, s_i=0.7):
        # TO BE MORE COMPETABILE TO THE MODEL
        return np.random.rand(len(age_group)) > s_i

    def _dummy_daily_duration(self):
        return (
            self.task_conf["threshold"]
            * self.task_conf["D_max"]
            / self.task_conf["P_max"]
            + self.task_conf["D_min"]
        )

    def _is_enough_duration(self, daily_duration):
        return np.where(
            daily_duration.values >= self.task_conf.get("D_min"),
            daily_duration.values
            / self.task_conf.get("D_max")
            * self.task_conf.get("P_max"),
            0,
        ) > self.task_conf.get("threshold")

    def _final_state(self, color: bool):
        if color:
            P_r = np.random.normal(*[0.08, 0.03])
            return np.random.choice(["w", "k"], 1, p=[1 - abs(P_r), abs(P_r)]).item()
        else:
            return "w"

    def _infection_state_transition(self, infected, infection_date):
        # get info about an infected person and return relevant data for dataframe
        df = pd.DataFrame(
            np.random.choice(
                len(self.task_conf.get("age_dist")),
                len(infected.index),
                p=self.task_conf.get("age_dist"),
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

    def get_trajectory(self, infected, output, curr_date, add_duration=True):
        if isinstance(infected, set):
            infected = pd.DataFrame(index=infected)
            if add_duration:
                infected["daily_duration"] = self._dummy_daily_duration()  # = D_i
        # remove newly infected if they've already been infected in the past
        infected = infected.loc[~infected.index.isin(output.df.index)]
        infected = infected[self._is_enough_duration(infected)]
        # no need to add anything if there are no newly infected today
        if infected.empty:
            return
        infected = infected.join(self._infection_state_transition(infected, curr_date))
        output.append(infected)
        return infected
