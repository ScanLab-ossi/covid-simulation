from functools import wraps
from time import time
from timeit import timeit


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


def timing_loop(f):
    @wraps(f)
    def wrap(*args, **kw):
        timeit(f(*args, **kw), number=10)

    return
