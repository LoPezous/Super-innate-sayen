import random
import pandas as pd
import numpy as np
import re
from FlowCytometryTools import FCMeasurement
import profiler
from more_itertools import one


@profiler.sayen_logger
@profiler.sayen_timer
def traitement(baseline_files, files, animals, cells, scaled=True):
    for file in files:
        if re.match('BL', file):
            baseline_files.append(file)

    sample_BL = FCMeasurement(ID='Test Sample', datafile=f"files/{baseline_files[0]}").data
    for animal in animals:
        if animal in baseline_files[0]:
            sample_BL['animal'] = [animal] * len(sample_BL)

    baseline_files.pop(0)

    for file in baseline_files:

        sample = FCMeasurement(ID='Test Sample', datafile=f"files/{file}")
        sample = sample.data
        for animal in animals:
            if animal in file:
                sample['animal'] = [animal] * len(sample)
        sample_BL.append(sample)
        del sample

    # Resampling BASELINE

    indexes = random.sample(range(0, len(sample_BL)), cells)
    data_BL = sample_BL
    del sample_BL
    data_BL = data_BL.iloc[indexes]

    columns_to_drop = ['Time', 'Event_length', 'Center', 'Offset', 'Width',
                       'Residual', 'FileNum', '102Pd', '103Rh', '104Pd',
                       '105Pd', '106Pd', '108Pd', '110Pd', '190BCKG',
                       '191Ir', '193Ir', '80ArAr', '131Xe_conta', '140Ce_Beads',
                       '208Pb_Conta', '127I_Conta', '138Ba_Conta']
    # Cleaning BASELINE
    data_BL = data_BL.drop(columns_to_drop, axis=1)

    data_BL['Timepoint'] = ['BL'] * len(data_BL)  # ADD A TIMEPOINT COLUMN

    # ALL D28 files

    matches = list()

    variables = dict()

    timepoints = data_BL['Timepoint']
    animales = data_BL['animal']

    del data_BL['Timepoint']
    del data_BL['animal']
    data = data_BL

    for file in files:
        findall = re.findall(r'(D\d{2,3}_)', file)
        if len(findall) == 1:
            matches.append(one(findall))

    matches = np.unique(matches)
    for match in matches:
        match = match.replace('_', '')
        timepoint_files = list()
        for file in files:
            if re.search(match, str(file)):
                timepoint_files.append(file)

        # match = 'sample_' + str(match)
        key = match
        variables[key] = FCMeasurement(ID='Test Sample', datafile=r'files/' + timepoint_files[0]).data
        # variables[key] = value.data
        for animal in animals:
            if animal in timepoint_files[0]:
                variables[key]['animal'] = [animal] * len(variables[key])

        timepoint_files.pop(0)
        for item in timepoint_files:
            sample = FCMeasurement(ID='Test Sample', datafile=f"files/{file}")
            sample = sample.data
            for animal in animals:
                if animal in item:
                    sample['animal'] = [animal] * len(sample)
            variables[key] = variables[key].append(sample)
            del sample

        # indexes = random.sample(range(0, len(variables[key])), cells)
        indexes = random.sample(range(0, len(variables[key])), cells)

        variables[key] = variables[key].iloc[indexes]

        # cleaning D28

        variables[key] = variables[key].drop(columns_to_drop, axis=1)

        variables[key]['Timepoint'] = [match] * len(variables[key])

        # RESAMPLING D28

        # ADD A TIMEPOINT COLUMN

        # ADD OTHER TIMEPOINT CODE IF NECESSARY HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # SECTIONS TO COPY AND CHANGE: ALL {timepoint} files, Resampling {timepoint},  cleaning {timepoint}

        # CREATE timepoint DF
        timepoints = timepoints.append(variables[key]['Timepoint'], ignore_index=True)  # serie pandas
        animales = animales.append(variables[key]['animal'], ignore_index=True)         # serie pandas
        del variables[key]['Timepoint']
        del variables[key]['animal']

        # DELETE TIMEPOINTS FROM ANALYSIS DF (if other timepoints were added, also delete timepoint column from these
        # dataframes)

        # CREATE ANALYSIS DATAFRAME with same indices as timepoint df

        data = data.append(variables[key], ignore_index=True)

    data.columns = ['CD45', 'CD66', 'HLA-DR', 'CD3',
                    'CD64', 'CD34', 'H3', 'CD123', 'CD101',
                    'CD38', 'CD2', 'Ki67', 'CD10', 'CD117',
                    'CX3CR1', 'E3L', 'CD172a', 'CD45RA',
                    'CD14', 'Siglec1', 'CD1C', 'H4K20me3',
                    'CD32', 'CLEC12A', 'CD90', 'H3K27ac', 'CD16',
                    'CD11C', 'CD33', 'H4', 'CD115', 'BDCA2', 'CD49d+',
                    'H3K27me3', 'H3K4me3', 'CADM1', 'CD20', 'CD8', 'CD11b']  # 39d
    del data_BL
    # el data_D28
    # drop non-clustering markers and keep them for later use
    back_up = data
    data = data.drop(['Ki67', 'H3K4me3', 'H3K27me3', 'H4', 'H3K27ac', 'H4K20me3', 'E3L', 'CD64', 'CD2', 'CD45RA'],
                     axis=1)  # 32d 6 --> 29

    # DATA PER ANIMAL

    for animal in np.unique(animales):
        anml = (animales == animal)
        df = pd.DataFrame(data[anml])
        df.to_csv(f"{animal}.csv")

    data = np.arcsinh(data)
    if scaled:
        data = (data - data.min()) / (data.max() - data.min())  # MINMAX NORMED

    return data, back_up, timepoints, matches, animales
