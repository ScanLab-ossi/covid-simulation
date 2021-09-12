from functools import wraps
from datetime import datetime
import pickle
import sys
import logging

# logging.basicConfig(format=FORMAT)

from simulation.constants import *


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = datetime.now()
        result = f(*args, **kw)
        te = datetime.now()
        if settings["VERBOSE"] or f.__name__ == "contagion_runner":
            print(f"func: {f.__name__} took: {te-ts}")
            logging.info(f"func: {f.__name__} took: {te-ts}")

        # print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def print_settings(task):
    print(f"DATASET = {task['DATASET']}")
    print(f"ITERATIONS = {task['ITERATIONS']}")
    print(f"SENSITIVITY = {task['SENSITIVITY']}")
    print(*(f"{k} = {v}" for k, v in settings.items()), sep="\n")


def increment():
    if settings["INCREMENT"]:
        if input("continue? y/(n)") != "y":
            sys.exit()
