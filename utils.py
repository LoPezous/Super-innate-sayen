""" Python file containing some utils functions that are called in several other main scripts."""
import time
import numpy as np
import scipy.stats as stat
from unidip import UniDip
from file_params import INFO_PATH


def unimodal(dat):
    dat = list(dat)
    dat = np.msort(dat)
    intervals = UniDip(dat, alpha=0.05).run()
    return intervals


def spread(dat):
    return stat.iqr(dat)


def unimodal_bool(dat):
    dat = list(dat)  # TODO : Essaye np.asarray()
    dat = np.msort(dat)
    intervals = UniDip(dat, alpha=0.05).run()
    if len(intervals) != 1:
        return False
    else:
        return True


def spread_bool(dat):
    IQR = stat.iqr(dat)
    if IQR < 200:
        return True
    else:
        return False


def write_info(text, kind="[INFO]"):
    with open(INFO_PATH, "a") as f:
        f.write(f"{kind} - {text} - ({time.asctime()}\n)")

