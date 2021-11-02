""" File containing all the constant variables where the path for output and input files are set.
    Some functions for creating files can be found as well
"""
import os
import re
import shutil
import sys
from FlowCytometryTools import FCMeasurement

FILES_PATH = "C:\\Users\\mp268043\\Jupyter\\tests\\VAC2022\\contrôles\\files"  # Double \ for Windows
BL_PATH = "C:\\Users\\mp268043\\Jupyter\\tests\\VAC2022\\contrôles\\files\\BL"
D28_PATH = "C:\\Users\\mp268043\\Jupyter\\tests\\VAC2022\\contrôles\\files\\D28"

# Path for logging
INFO_PATH = "info"
RESULTS_PATH = "results"
cwd = sys.path[0]


def get_files(metric, cwd_=cwd):
    os.chdir(cwd_)

    paths = ['Results', 'Results/metrics', f"Results/metrics/{metric}",
             f'Results/metrics/{metric}/_1', f'Results/metrics/{metric}/_1/UMAP',
             f'Results/metrics/{metric}/_1/UMAP/other_markers',
             f'Results/metrics/{metric}/_1/Clusters',
             f'Results/metrics/{metric}/_01', f'Results/metrics/{metric}/_01/UMAP',
             f'Results/metrics/{metric}/_01/UMAP/other_markers',
             f'Results/metrics/{metric}/_01/Clusters']

    for path in paths:

        if os.path.exists(path):  # DELETES EXISTING DIRS AND RECREATE THEM
            shutil.rmtree(path)
        os.makedirs(path)

    # BASELINE
    files = os.listdir('files')

    # ALL BASELINE FILES

    baseline_files = list()
    for file in files:
        if re.match('BL', file):
            baseline_files.append(file)
    return baseline_files, files


def open_FCS(path, ID):
    file = FCMeasurement(ID=ID, datafile=path)
    data = file.data
    file.close()  # TODO: Vérifier si on peut fermer un FCMeasurement
    return data
