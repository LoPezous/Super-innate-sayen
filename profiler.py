""" Module containing methods to profile the code.
Ex : Keep track of memory or the time execution.
"""

import logging
import time
from functools import wraps

# sys.getsizeof(variable)  # to get the size in bytes of a variable


# Examples adapted from Corey Schafer's tutorial on decorators
def sayen_logger(func):
    """ Function writing information on how a function has been called. """
    logging.basicConfig(filename=f"{func.__name__}_log", level=logging.INFO)

    @wraps(func)  # If we want to chain decorators
    def wrapper(*args, **kwargs):
        logging.info(f"Ran with args : {args}, and kwargs: {kwargs}")  # TODO: Ajouter l'heure d'exécution
        return func(*args, **kwargs)

    return wrapper


def sayen_timer(func):
    logging.basicConfig(filename=f"{func.__name__}_time", level=logging.INFO)

    @wraps(func)  # If we want to chain decorators
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        res = func(*args, **kwargs)
        t2 = time.perf_counter() - t1
        logging.info(f"Ran in {t2} seconds")
        return res

    return wrapper

# TODO: Add time and memory loggings
# Try not to print anything to the screen ?
# The sayen_logger is now pretty useless but can be used as a template to create useful logging information


""" Other example (PyCon 2019) """


def logtime(func):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        res = func(*args, **kwargs)
        t2 = time.perf_counter() - t1

        with open("time_perf.log", "a") as out:
            out.write(f"{time.asctime()}\t{func.__name__}\t{t2}\n")
        return res
    return wrapper
