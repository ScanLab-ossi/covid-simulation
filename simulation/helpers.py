from functools import wraps
from datetime import datetime
import sys
import logging
from yaml import dump

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


def increment():
    if settings["INCREMENT"]:
        if input("continue? y/(n)") != "y":
            sys.exit()
