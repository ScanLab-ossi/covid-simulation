from random import seed
from random import randint
from random import choices
import random
import pickle
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import psycopg2 as psycopg2

conf = {
    "age_dist": np.array([0.15, 0.6, 0.25]),  # [youngs, , adults, olds]
    "recovery_time_dist": np.array([20, 10]),  # recovery_time_dist ~ Norm(mean, std) | ref:
    "aggravation_time_dist": np.array([5, 2]),  # aggravation_time_dist ~ Norm(mean, std) | ref:
    "D_min": 10,  # Arbitrary, The minimal threshold (in time) for infection,
    "number_of_patient_zero": 10,  # Arbitrary
    "D_max": 70,  # Arbitrary, TO BE CALCULATED,  0.9 precentile of (D_i)'s
    "P_max": 0.2,  # The probability to be infected when the exposure is over the threshold
    "risk_factor": None  # should be vector of risk by age group
}


class Config(object):
    def __init__(self, config_file):
        self._config = config_file  # set it to conf

    def get_property(self, property_name):
        if property_name not in self._config.keys():  # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]


class Data(object):
    def __init__(self):
        self.df = pd.DataFrame(columns=['id', 'age_group', 'color', 'infection_date', 'expiration_date'])

    def display(self):
        print(self.df.to_string())

    def shape(self):
        print(self.df.shape)

    def export(self):
        self.df.to_csv('output.csv', index=False)

    def append(self, id, age_group, infection_date, expiration_date, color):
        self.df = self.df.append({'id': id, 'age_group': age_group, 'infection_date': infection_date,
                                  'expiration_date': expiration_date, "color": color}, ignore_index=True)

    @staticmethod
    def time_to_recovery(infection_date, config):
        mu, sigma = config.get_property("recovery_time_dist")[0], config.get_property("recovery_time_dist")[1]
        recovery_duration = int(np.around(np.random.normal(mu, sigma)))
        if recovery_duration <= 1:  # Avoid the paradox of negative recovery duration.
            recovery_duration = 1
        expiration_date = datetime.strptime(infection_date, '%Y-%m-%d') + timedelta(recovery_duration)
        return expiration_date.strftime('%Y-%m-%d')

    @staticmethod
    def time_to_aggravation(infection_date, config):
        mu, sigma = config.get_property("aggravation_time_dist")[0], config.get_property("aggravation_time_dist")[1]
        aggravation_duration = int(np.around(np.random.normal(mu, sigma)))
        if aggravation_duration <= 1:  # Avoid the paradox of negative recovery duration.
            aggravation_duration = 1
        expiration_date = datetime.strptime(infection_date, '%Y-%m-%d') + timedelta(aggravation_duration)
        return expiration_date.strftime('%Y-%m-%d')

    @staticmethod
    def check_if_aggravate(age_group=None, s_i=0.7):
        # TO BE MORE COMPETABILE TO THE MODEL
        threshold = s_i
        prob = np.random.rand(1)
        return (prob > threshold)

    @staticmethod
    def check_if_infected(P_max, P_gb, threshold=0.05):
        return P_max * P_gb > threshold

    @staticmethod
    def infection_state_transition(config, infection_date):
        # get info about an infected person and return relevant data for dataframe
        age_dist = config.get_property("age_dist")
        age_group = choices(np.arange(len(age_dist)), age_dist)
        # print(age_dist, np.arange(len(age_dist)), age_group)
        if (Data.check_if_aggravate(age_group=age_group)):
            color = "purple"
            expiration_date = Data.time_to_aggravation(infection_date, config)
        else:
            color = "blue"
            expiration_date = Data.time_to_recovery(infection_date, config)
        return age_group, color, expiration_date


def pick_patient_zero(set_of_potential_patients, num_of_patients=1, random_seed=1, arbitrary=False,
                      arbitrary_patient_zero=["MJviZSTPuYw1v0W0cURthY"]):
    # return set of zero patients
    if arbitrary:
        return arbitrary_patient_zero
    else:
        seed(random_seed)
        randomly_patient_zero = random.sample(set_of_potential_patients, num_of_patients)
        return randomly_patient_zero


def get_active_ids(data):  # return all the people who make contact in the given dataset (data is csv or sql query)
    active_ids = set()
    with open(data) as fp:
        for line in fp:
            active_ids.add(line.split(",")[0])
            active_ids.add(line.split(",")[5])
    return active_ids


def contagion_in_csv(data_as_csv_file, infected_list, date):
    contagion_set = set()
    with open(data_as_csv_file) as fp:
        for line in fp:
            if line.split(",")[2] == date:
                if line.split(",")[0] in infected_list:
                    contagion_set.add(line.split(",")[5])
                if line.split(",")[5] in infected_list:
                    contagion_set.add(line.split(",")[0])

    return contagion_set


def contagion_in_sql(infected_set, date):
    if len(infected_set) == 0:  # empty set
        return set()
    # data format is "YYYY-MM-DD"
    contagion_set = set()
    # Connect to an existing database
    conn = psycopg2.connect(dbname="DB", user="user", password="pass", host="1")
    # Open a cursor to perform database operations
    cur = conn.cursor()
    infected_tuple = str(tuple(infected_set))
    date_as_str = "'" + str(date) + "'"
    # Query the database and obtain data as Python objects
    query = f"""select  destination as a
                   from h3g.call
                   where date between {date_as_str} and {date_as_str}
                   AND
                   source in {infected_tuple}
                   Union
                   select source as a
                   from h3g.call
                   where date between {date_as_str} and {date_as_str}
                   AND
                   destination in {infected_tuple};"""
    print(len(query))
    cur.execute(query)
    cur_out_as_arr = np.asarray(cur.fetchall())
    contagion_set = set(cur_out_as_arr.flatten())
    # Close communication with the database
    cur.close()
    conn.close()
    return contagion_set


def daily_duration_in_sql(id, date):
    # to be completed
    return 50


def virus_spread(data, set_of_patients, start_date, days):
    date_str = start_date
    date_object = datetime.strptime(date_str, '%Y-%m-%d')
    patients = set_of_patients
    for i in range(days):
        new_patients = contagion_in_csv(data, patients, date_object.strftime('%Y-%m-%d'))
        patients = patients.union(new_patients)
        date_object += timedelta(days=1)
    return patients


def one_array_pickle_to_set(pickle_file_name):
    # open a file, where you stored the pickled data
    file = open(pickle_file_name, 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()
    set_from_arr = set(data.flatten())
    return set_from_arr


def contagion_runner(config):
    config = Config()
    data = Data()
    period = 10  # max is 65
    first_date = date(2012, 3, 26)  #
    pickle_of_ids = one_array_pickle_to_set('destination_ids_first_3days.pickle')
    zero_patients = pick_patient_zero(pickle_of_ids, num_of_patients=config.get_property("number_of_patient_zero"))
    # -- TO DO -- append the zero patients into dataframe
    for id in zero_patients:
        age_group, color, expiration_date = data.infection_state_transition(config, first_date.strftime('%Y-%m-%d'))
        data.append(id, age_group, first_date.strftime('%Y-%m-%d'), expiration_date, color)

    set_of_infected = set(zero_patients)
    for day in range(period):
        curr_date = first_date + timedelta(days=day)
        curr_date_as_str = curr_date.strftime('%Y-%m-%d')
        set_of_contact_with_patient = contagion_in_sql(set_of_infected, curr_date_as_str)
        for id in set_of_contact_with_patient:
            daily_duration = daily_duration_in_sql(id, curr_date_as_str)  # = D_i
            P_gb = daily_duration / config.get_property("D_max") if daily_duration > config.get_property("D_min") else 0
            P_max = config.get_property("P_max")
            if data.check_if_infected(P_max, P_gb):
                age_group, color, expiration_date = data.infection_state_transition(config, curr_date_as_str)
                data.append(id, age_group, curr_date_as_str, expiration_date, color)

        set_of_infected = set(
            data.df[(data.df['expiration_date'] > curr_date_as_str)]["id"].values)
        # patients that doesn't recovered or died yet

        print(f"Status of {day}:")
        data.display()

    data.export()
    print("success")


if __name__ == '__main__':
    contagion_runner(conf)
