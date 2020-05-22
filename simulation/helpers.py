from functools import wraps
from time import time
import pickle


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func: %r took: %2.4f sec" % (f.__name__, te - ts))
        # print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


def one_array_pickle_to_set(pickle_file_name):
    # open a file, where you stored the pickled data
    with open(pickle_file_name, "rb") as f:
        data = pickle.load(f)
    set_from_arr = set(data.flatten())
    return set_from_arr
