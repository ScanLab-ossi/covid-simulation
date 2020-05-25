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

    def expiration_date(self, color, infection_date):
        time_dist = "recovery_time_dist" if color else "aggravation_time_dist"
        duration = int(np.around(np.random.normal(*self.task_conf.get(time_dist))))
        if duration <= 1:  # Avoid the paradox of negative recovery duration.
            duration = 1
        expiration_date = infection_date + timedelta(duration)
        if expiration_date > self.dataset.end_date:
            return self.dataset.end_date
        return expiration_date

    def check_if_aggravate(self, age_group, s_i=0.7):
        # TO BE MORE COMPETABILE TO THE MODEL
        return np.random.rand(len(age_group)) > s_i

    def daily_duration_in_sql(self, id=None, date=None):
        # to be completed
        return 90

    def is_enough_duration(self, daily_duration):
        return (
            np.where(
                daily_duration.values >= self.task_conf.get("D_min"),
                daily_duration.values / self.task_conf.get("D_max"),
                0,
            )
            * self.task_conf.get("P_max")
            > 0.05
        )

    def infection_state_transition(self, infected, infection_date):
        # get info about an infected person and return relevant data for dataframe
        df = pd.DataFrame(
            np.random.choice(
                len(self.task_conf.get("age_dist")),
                len(infected.index),
                p=self.task_conf.get("age_dist").tolist(),
            ),
            columns=["age_group"],
            index=infected.index,
        )
        # True=purple, False=blue
        df["color"] = self.check_if_aggravate(df["age_group"].values)
        df["expiration_date"] = df["color"].apply(
            self.expiration_date, args=(infection_date,)
        )
        return df

    @timing
    def get_trajectory(self, infected, output, curr_date, add_duration=True):
        if isinstance(infected, set):
            infected = pd.DataFrame(index=infected)
            if add_duration:
                infected["daily_duration"] = daily_duration_in_sql()  # = D_i
        # remove newly infected if they've already been infected in the past
        infected = infected.loc[~infected.index.isin(output.df.index)]
        if "daily_duration" in infected.columns:
            infected = infected[self.is_enough_duration(infected)]
        infected = infected.join(self.infection_state_transition(infected, curr_date))
        infected["infection_date"] = curr_date
        output.append(infected)
